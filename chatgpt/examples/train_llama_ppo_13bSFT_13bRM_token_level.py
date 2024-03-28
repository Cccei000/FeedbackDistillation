# encoding=utf-8

"""Train"""
import os
import torch
import torch.distributed.run
import argparse

from fengshen_inner.models.model_utils import add_module_args, add_inverse_square_args
from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM
from fengshen_inner.models.megatron import mpu

from chatgpt.dataset import ReplayBufferDataset
from chatgpt.nn.llama import LlamaActor, LlamaCritic, modeling_fengshenLlama_rm,modeling_fengshenLlama_critic
from chatgpt.trainer import PPOTrainer
from chatgpt.experience_maker import LocalInferExperienceMaker
from chatgpt.replay_buffer import NaiveReplayBuffer, DistributedBatchSampler
from chatgpt.logger import WandbLogger
from chatgpt.utils import local_rank, print_rank_0
from chatgpt.strategies import add_megatron_deepspeed_args, initialize_megatron, build_deepspeed_config, setup_model_and_optimizer, get_save_checkpoint_callback
from chatgpt.nn.utils import zero_pad_sequences
from transformers import LlamaTokenizer, AutoTokenizer
from tokenizers import AddedToken

_POLICY_TOKENIZER_PATH = "/cognitive_comp/songzhuoyang/models/llama_sft/20230405v1"
_PRETRAIN_MODEL_PATH = '/cognitive_comp/wanghao/models/llama_sft/13b_0423_MP2'
_REWARD_MODEL_PATH = '/cognitive_comp/liangyuxin/workspace/rm_train/RM_0412_mix_7B/ckpt/last.ckpt/checkpoint/mp_rank_00_model_states.pt'
_REWARD_CONFIG_PATH = '/cognitive_comp/sunqianguo/pretrained/checkpoints/7B/0405/v2/checkpoint-16000/config.json'
_PPO_DATASET_PATH = '/cognitive_comp/songzhuoyang/processed_data/mixed_ppo_dataset_0327_for_llama'


_SPECIAL_TOKENS_DICT = {'pad_token': '</s>'}
human = AddedToken("<human>",lstrip=False,rstrip=False,single_word=False,normalized=True)
bot = AddedToken("<bot>",lstrip=False,rstrip=False,single_word=False,normalized=True)

def add_neox_ppo_pipeline_args(parent_args:argparse.ArgumentParser):
    
    group = parent_args.add_argument_group("Experiment Args")
    
    group.add_argument("--wandb_project", type=str, default="PPO_LLAMA")
    group.add_argument("--wandb_group", type=str, default="")
    group.add_argument("--wandb_team", type=str, default=None)
    group.add_argument("--wandb_name", type=str, default=None)
     
    group = parent_args.add_argument_group("PPO Args")
    group.add_argument("--num_episodes", type=int, default=1, help="训练轮数，每轮中包括经验池采样、奖励模型打分、模型训练三步")
    group.add_argument("--max_timesteps", type=int, default=1, help="每轮中进行经验池采样的次数")
    group.add_argument("--update_timesteps", type=int, default=1, help="")
    group.add_argument("--sample_replay_buffer", action="store_true", default=False)
    group.add_argument("--sample_batch_size", type=int, default=32, help="每次经验池采样中，使用的Prompt数量（不考虑数据并行）")
    group.add_argument("--buffer_limit_size", type=int, default=512)
    group.add_argument("--max_epoch_per_update", type=int, default=2, help="每次模型训练时，训练的epoch数")
    group.add_argument("--replay_buffer_cpu_offload", type=bool, default=True)
    group.add_argument("--entropy_loss_coef", type=float, default=0.01)
    group.add_argument("--entropy_loss_decay_rate", type=float, default=0.98)
    group.add_argument("--clip_grad", action="store_true", default=False)
    
    group.add_argument("--eps_clip", type=float, default=0.2)
    group.add_argument("--value_clip", type=float, default=0.2)
    
    group.add_argument("--enable_gae", action="store_true", default=False)
    group.add_argument("--gamma", type=float, default=1.0)
    group.add_argument("--lam", type=float, default=0.95)
    group.add_argument("--token_level_reward", action="store_true", default=False)
    group.add_argument("--enable_reward_scaling", action="store_true", default=False)
    group.add_argument("--enable_token_level_loss", action="store_true", default=False)
    group.add_argument("--mix_reward", action="store_true", default=False)
    
    group = parent_args.add_argument_group("Experience Args")
    group.add_argument("--top_p", type=float, default=0.85)
    group.add_argument("--top_k", type=int, default=0)
    group.add_argument("--kl_coef", type=float, default=0.0)
    group.add_argument("--max_length", type=int, default=1024)
    group.add_argument("--max_new_tokens", type=int, default=512)
    group.add_argument("--repetition_penalty", type=float, default=1.)
    group.add_argument("--temperature", type=float, default=1.)
    group.add_argument("--experience_batch_size", type=int, default=32)
    group.add_argument("--policy_minibatch_size", type=int, default=4)
    group.add_argument("--gen_minibatch_size", type=int, default=4)
    group.add_argument("--rm_minibatch_size", type=int, default=1)
    group.add_argument("--rm_model_max_seq_len", type=int, default=1024)
    group.add_argument("--prompt_dataset_path", type=str, default=_PPO_DATASET_PATH, help="用于训练的所有prompt") # 格式参考默认路径的dataset
    group.add_argument("--exp_save_path", type=str, default="/cognitive_comp/songzhuoyang/workspace/chatgpt/6B_rlhf/exp", help="训练产生的经验池的保存路径") 
    
    group = parent_args.add_argument_group("Trainer Args")
    group.add_argument("--num_workers", type=int, default=2)
    group.add_argument("--total_steps", type=int, default=1e4)
    group.add_argument("--policy_train_batch_size", type=int, default=1)
    group.add_argument("--do_validation", action="store_true", default=False)
    group.add_argument("--val_check_interval", type=int, default=5)
    group.add_argument("--val_size_per_task", type=int, default=20)
    group.add_argument("--actor_lr", type=float, default=2e-6)
    group.add_argument("--critic_lr", type=float, default=1e-5)
    
    
    group = parent_args.add_argument_group("Model Args")
    group.add_argument("--policy_ckpt_path", type=str, default="/cognitive_comp/songzhuoyang/workspace/chatgpt/7B_rlhf/ckpt", help="rlhf ckpt保存的根目录") # 训练过程中会自动创建文件夹，保存每个episode的ckpt
    group.add_argument("--policy_tokenizer_path", type=str, default=_POLICY_TOKENIZER_PATH) # tokenizer路径
    group.add_argument("--policy_model_path", type=str, default=_PRETRAIN_MODEL_PATH, help="生成模型sft ckpt路径") # 需要使用gpt-neox/utils中的gxy写的转换脚本转换参数的key
    group.add_argument("--rm_config_path", type=str, default=_REWARD_CONFIG_PATH, help="rm config路径") # 训练过程中会自动创建文件夹，保存每个episode的ckpt
    group.add_argument("--rm_model_path", type=str, default=_REWARD_MODEL_PATH, help="rm ckpt路径") # 需要使用gpt-neox/utils中的gxy写的转换脚本转换参数的key
    group.add_argument("--rm_model_tokenizer_path", type=str, default=_POLICY_TOKENIZER_PATH) # tokenizer路径
    # group.add_argument("--ref_model_path", type=str, default=_PRETRAIN_MODEL_PATH, help="reference ckpt路径") # 需要使用gpt-neox/utils中的gxy写的转换脚本转换参数的key

    
    
    return parent_args
     

def get_dataloader_build_func(args, tokenizer:LlamaTokenizer):
    def build_dataloader(replay_buffer, episode, timestep):
        
        dataset = ReplayBufferDataset(
            replay_buffer, pad_token_id=tokenizer.pad_token_id, enable_gae=args.enable_gae, gamma=args.gamma, lam=args.lam
        )
        
        import datasets as ds
        gen_data = []
        reward = []
        ret = []
        sequence = []
        action_logprob = []
        attn_mask = []
        action_mask = []
        adv = []
        values = []
        
        for item in dataset:
            text = tokenizer.decode(item.sequences, skip_special_tokens=False)
            text = text.replace('<s>','').replace('</s>', '')

            gen_data.append(text)
            sequence.append(item.sequences.tolist())
            action_logprob.append(item.action_log_probs.tolist())
            attn_mask.append(item.attention_mask.tolist())
            action_mask.append(item.action_mask.tolist())
            if args.enable_gae:
                adv.append(item.advantages.tolist())
                reward.append(item.origin_reward.tolist())
                ret.append(item.reward.tolist())
                values.append(item.values.tolist())
            else:
                adv.append(item.advantages.item())
                reward.append(item.origin_reward.item())
                ret.append(item.reward.tolist())
                values.append(item.values.item())
        
        hf_dataset = ds.Dataset.from_dict({
            "item": gen_data,
            "reward": reward,
            "return": ret,
            "values": values,
            "advantage": adv,
            "sequence": sequence,
            "action_logprob": action_logprob,
            "attn_mask": attn_mask,
            "action_mask": action_mask
        })

        rank = mpu.get_data_parallel_rank()
        mp_rank = mpu.get_model_parallel_rank()
        pp_rank = mpu.get_pipe_parallel_rank()
        world_size = mpu.get_data_parallel_world_size()
        global_batch_size = args.policy_train_batch_size * world_size

        if mp_rank == 0 and pp_rank == 0:
            hf_dataset.save_to_disk(os.path.join(args.exp_save_path, f"{args.wandb_project}_ep{str(episode).zfill(3)}_{str(timestep).zfill(3)}_{rank}/"))

        # print(f"device:{example.sequences.device} - {example.reward.item()} - {text}")
            print(f"adv_mean:{dataset.collate_fn.adv_mean} | adv_std:{1 / dataset.collate_fn.inverse_adv_std}")


        # Use a simple sampler with distributed batch sampler
        sampler = torch.utils.data.SequentialSampler(dataset)
        batch_sampler = DistributedBatchSampler(
            sampler=sampler,
            batch_size=global_batch_size,
            drop_last=True,
            rank=rank,
            world_size=world_size,
        )

        # Torch dataloader
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
    return build_dataloader

def get_experience_convert_func(args, src_tokenizer:LlamaTokenizer, dst_tokenizer:LlamaTokenizer):
    
    def convert_func(sequences:torch.Tensor, attention_mask:torch.Tensor, action_mask:torch.Tensor, device:torch.device):

        last_attn_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in attention_mask], dtype=torch.int64)
        last_act_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in action_mask], dtype=torch.int64)
        
        end_with_eos = torch.diagonal(sequences[:,last_attn_index] == src_tokenizer.eos_token_id)
        
        new_attn_msk = attention_mask.clone().detach()
        new_act_msk = action_mask.clone().detach()

        for idx in range(sequences.shape[0]):
            if end_with_eos[idx]:
                new_attn_msk[idx, last_attn_index[idx]] = False
                new_act_msk[idx, last_act_index[idx]] = False
            else:
                text = src_tokenizer.decode(sequences[idx]).replace("\n", "\\n").replace("\t", "\\t")
                print_rank_0(f"TRUNC:{text}")
        
        return sequences, new_attn_msk, new_act_msk
    
    return convert_func


def get_val_experience_saving_func(args, tokenizer:LlamaTokenizer):
    def save_val_experience(replay_buffer, episode, task_name):
        dataset = ReplayBufferDataset(replay_buffer, pad_token_id=tokenizer.pad_token_id, enable_gae=args.enable_gae, gamma=args.gamma, lam=args.lam)
        
        import datasets as ds
        gen_data = []
        reward = []
        ret = []
        sequence = []
        action_logprob = []
        attn_mask = []
        action_mask = []
        adv = []
        values = []
        
        for item in dataset:
            text = tokenizer.decode(item.sequences, skip_special_tokens=False)
            text = text.replace('<s>','').replace('</s>', '')
            gen_data.append(text)
            sequence.append(item.sequences.tolist())
            action_logprob.append(item.action_log_probs.tolist())
            attn_mask.append(item.attention_mask.tolist())
            action_mask.append(item.action_mask.tolist())
            if args.enable_gae:
                adv.append(item.advantages.tolist())
                reward.append(item.origin_reward.tolist())
                ret.append(item.reward.tolist())
                values.append(item.values.tolist())
            else:
                adv.append(item.advantages.item())
                reward.append(item.origin_reward.item())
                ret.append(item.reward.item())
                values.append(item.values.tolist())
            
        assert len(task_name) == len(gen_data)
        
        hf_dataset = ds.Dataset.from_dict({
            "item": gen_data,
            "task": task_name,
            "reward": reward,
            "return": ret,
            "values": values,
            "advantage": adv,
            "sequence": sequence,
            "action_logprob": action_logprob,
            "attn_mask": attn_mask,
            "action_mask": action_mask
        })

        rank = mpu.get_data_parallel_rank()
        mp_rank = mpu.get_model_parallel_rank()
        pp_rank = mpu.get_pipe_parallel_rank()
        if mp_rank == 0 and pp_rank == 0:
            hf_dataset.save_to_disk(os.path.join(args.exp_save_path, f"{args.wandb_project}_val_ep{str(episode).zfill(3)}_{rank}/"))
        return
    return save_val_experience
        

def load_dateset(args):
    
    import datasets as ds
    import random
    from collections import defaultdict
    ds.disable_caching()
    dataset = ds.load_from_disk(args.prompt_dataset_path)
    prompts = list(dataset["query"])
    
    if "task" not in dataset.column_names or not args.do_validation:
        return prompts, None
    
    tasks = list(dataset["task"])
    task_to_prompts = defaultdict(list)
    for task_name, query in zip(tasks, prompts):
        task_to_prompts[task_name].append(query)
    
    train_queries = []
    val_queries = {}
    for task_name, queries in task_to_prompts.items():
        g = torch.Generator()                                                                           
        g.manual_seed(args.seed + mpu.get_data_parallel_rank())                                                                       
        random_idx = torch.randperm(len(queries), generator=g).tolist()
        queries = [queries[i] for i in random_idx]
        # random.shuffle(queries)
        train_queries.extend(queries[:-args.val_size_per_task])
        val_queries[task_name] = queries[-args.val_size_per_task:]
    
        # print(f'task {task_name}, mp_rank {mpu.get_model_parallel_rank()}, data_rank {mpu.get_data_parallel_rank()}, queries {queries[:5]}', flush=True)
    
    return train_queries, val_queries

def launch(args):
    """Main training program.

    """
    # 初始化logger
    logger = WandbLogger(
        project=args.wandb_project,
        group=args.wandb_group,
        entity=args.wandb_team,
        ignore=local_rank() != 0,
        name=args.wandb_name,
    )
    logger.log_hyperparams(args)
    
    initialize_megatron(args)
    
    strategy = build_deepspeed_config(args)


    llama_tokenizer = AutoTokenizer.from_pretrained(args.policy_tokenizer_path,use_fast=False)
    llama_tokenizer.add_special_tokens(_SPECIAL_TOKENS_DICT)
    llama_tokenizer.add_special_tokens({"additional_special_tokens":[human, bot]})

    # print(f"ref model {args.policy_model_path}/part_{mpu.get_model_parallel_rank()}")
    # initial model
    im = LlamaActor(
        model=LlamaForCausalLM.from_pretrained(
            f"{args.policy_model_path}/part_{mpu.get_model_parallel_rank()}",
            # f"{args.ref_model_path}",
            torch_dtype=torch.bfloat16
        )
    ).eval().to(dtype=torch.bfloat16).cpu() #.to(dtype=torch.bfloat16)
    
    # reward model
    reward_model = modeling_fengshenLlama_rm(
        pretrained_path=f"{args.rm_model_path}/part_{mpu.get_model_parallel_rank()}",
        token_level_reward=args.token_level_reward,
        mix_reward=args.mix_reward, #是否使用token-sample-mix的RM
    ).eval().half().cpu()
    # deepspeed actor & critic
    actor = LlamaActor(
        model=LlamaForCausalLM.from_pretrained(
            f"{args.policy_model_path}/part_{mpu.get_model_parallel_rank()}",
            torch_dtype=torch.bfloat16
        )
    ).to(dtype=torch.bfloat16)
    # critic = LlamaCritic(
    #     model=LlamaForCausalLM.from_pretrained(
    #         f"{args.policy_model_path}/part_{mpu.get_model_parallel_rank()}",
    #         torch_dtype=torch.bfloat16
    #     ),
    #     return_mean=not args.enable_gae
    # ).to(dtype=torch.bfloat16)
    critic = modeling_fengshenLlama_critic(
        pretrained_path=f"{args.rm_model_path}/part_{mpu.get_model_parallel_rank()}",
        return_mean=not args.enable_gae
        ).to(dtype=torch.bfloat16)
    # print(critic.value_head.weight.data)
    
    actor, actor_optimizer, actor_lr = setup_model_and_optimizer(args, actor, strategy, args.actor_lr)
    critic, critic_optimizer, critic_lr = setup_model_and_optimizer(args, critic, strategy, args.critic_lr)

    # 初始化experience_maker replay_buffer

    experience_maker = LocalInferExperienceMaker(
        actor=actor, critic=critic, reward_model=reward_model, initial_model=im, kl_coef=args.kl_coef,
        seed=args.seed + mpu.get_data_parallel_rank(),
        pad_token_id=llama_tokenizer.pad_token_id,
        gen_minibatch_size=args.gen_minibatch_size,
        actor_minibatch_size=args.policy_minibatch_size,
        rm_minibatch_size=args.rm_minibatch_size,
        enable_gae=args.enable_gae,
        enable_reward_scaling=args.enable_reward_scaling,
        reward_scaling_gamma=args.gamma,
        token_level_reward=args.token_level_reward,
        logger=logger,
    )

    replay_buffer = NaiveReplayBuffer(
        sample_batch_size=args.sample_batch_size,
        limit=args.buffer_limit_size,
        cpu_offload=args.replay_buffer_cpu_offload,
    )


    tokenizer_vocab_size = llama_tokenizer.vocab_size
    policy_vocab_size = im.model.config.vocab_size


    bad_words_ids = [[llama_tokenizer.convert_tokens_to_ids(role)] for role in ['<human>','<bot>']]
    print_rank_0(f"Ignore all role tokens: bad_words_ids")

    if policy_vocab_size > tokenizer_vocab_size:
        bad_words_ids += [[ids] for ids in range(tokenizer_vocab_size, policy_vocab_size)]
        print_rank_0(f"BAD TOKEN IDS: {tokenizer_vocab_size}~{policy_vocab_size - 1}")

    # 初始化trainer
    generate_kwargs = {
        "do_sample": True,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "bad_words_ids": bad_words_ids,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "use_cache": True,
        "pad_token_id": llama_tokenizer.pad_token_id,
        "eos_token_id": llama_tokenizer.eos_token_id,
    }

    ppo_trainer = PPOTrainer(
        actor=actor,
        critic=critic,
        actor_optim=actor_optimizer,
        critic_optim=critic_optimizer,
        actor_lr_scheduler = actor_lr,
        critic_lr_scheduler = critic_lr,
        experience_maker=experience_maker,
        replay_buffer=replay_buffer,
        logger=logger,
        clip_grad=args.clip_grad,
        entropy_loss_coef = args.entropy_loss_coef,
        entropy_loss_decay_rate = args.entropy_loss_decay_rate,
        enable_token_level_loss=args.enable_token_level_loss,
        ckpt_saving_func=get_save_checkpoint_callback(args),
        experience_batch_size=args.experience_batch_size,
        setup_dataloader_func=get_dataloader_build_func(args, llama_tokenizer),
        eps_clip=args.eps_clip,                                # ratio 裁剪
        value_clip=args.value_clip,                            # value 裁剪
        max_epochs=args.max_epoch_per_update,                  # 每个训练阶段actor和critic的训练轮数
        tokenizer=llama_tokenizer,
        sample_replay_buffer=args.sample_replay_buffer,   # 每次使用全部经验池内容
        **generate_kwargs           # 模型生成样本的参数
    )

    train_prompts, val_prompts = load_dateset(args)
    
    if val_prompts is not None:
        print_rank_0(f"Train on {len(train_prompts)} prompts with validation every {args.val_check_interval} episode.")
    else:
        print_rank_0(f"Train on {len(train_prompts)} prompts without validation.")

    # 开始训练
    ppo_trainer.fit(
        prompts=train_prompts,
        val_prompts=val_prompts,
        seed=args.seed + mpu.get_data_parallel_rank(),
        val_check_interval=args.val_check_interval,
        val_saving_func=get_val_experience_saving_func(args, llama_tokenizer),
        num_episodes=args.num_episodes,
        max_timesteps=args.max_timesteps,     # 每一个episode采样经验的步数
        update_timesteps=args.update_timesteps,   # 训练模型的累积步数
    )
    return


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser = add_module_args(parent_args=parser)
    parser = add_inverse_square_args(parent_args=parser)
    parser = add_neox_ppo_pipeline_args(parent_args=parser)
    parser = add_megatron_deepspeed_args(parent_args=parser)
    
    args = parser.parse_args()
    
    launch(args)
    print("!")
