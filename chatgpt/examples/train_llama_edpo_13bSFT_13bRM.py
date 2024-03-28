# encoding=utf-8

"""Train"""
import os
import torch
import torch.distributed.run
import argparse

from fengshen_inner.models.model_utils import add_module_args, add_inverse_square_args
from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM
from fengshen_inner.models.megatron import mpu

from chatgpt.replay_buffer import PreferenceReplayBuffer, DistributedBatchSampler
from chatgpt.nn.llama import LlamaActor, modeling_fengshenLlama_rm
from chatgpt.trainer import EDPOTrainer
from chatgpt.experience_maker import EDPOExperienceMaker
from chatgpt.logger import WandbLogger
from chatgpt.utils import local_rank, print_rank_0
from chatgpt.strategies import add_megatron_deepspeed_args, initialize_megatron, build_deepspeed_config, setup_model_and_optimizer, get_save_checkpoint_callback
from chatgpt.nn.utils import zero_pad_sequences
from chatgpt.nn import TotGSArgs
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
    group.add_argument("--clip_grad", action="store_true", default=False)
            
    group = parent_args.add_argument_group("Experience Args")
    group.add_argument("--top_p", type=float, default=0.85)
    group.add_argument("--top_k", type=int, default=0)
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
    group.add_argument("--activation_checkpointing", action="store_true", default=False)

    
    group = parent_args.add_argument_group("Model Args")
    group.add_argument("--policy_ckpt_path", type=str, default="/cognitive_comp/songzhuoyang/workspace/chatgpt/7B_rlhf/ckpt", help="rlhf ckpt保存的根目录") # 训练过程中会自动创建文件夹，保存每个episode的ckpt
    group.add_argument("--policy_tokenizer_path", type=str, default=_POLICY_TOKENIZER_PATH) # tokenizer路径
    group.add_argument("--policy_model_path", type=str, default=_PRETRAIN_MODEL_PATH, help="生成模型sft ckpt路径") # 需要使用gpt-neox/utils中的gxy写的转换脚本转换参数的key
    group.add_argument("--rm_model_path", type=str, default=_REWARD_MODEL_PATH, help="rm ckpt路径") # 需要使用gpt-neox/utils中的gxy写的转换脚本转换参数的key
    group.add_argument("--rm_model_tokenizer_path", type=str, default=_POLICY_TOKENIZER_PATH) # tokenizer路径

    group = parent_args.add_argument_group("Tot Args")
    group.add_argument("--enabling_tot", action="store_true", default=False)
    group.add_argument("--gs_eval_batch_size", type=int, default=8)
    group.add_argument("--gs_gen_batch_size", type=int, default=8)
    group.add_argument("--gs_gen_repeat_times", type=int, default=2)
    group.add_argument("--gs_breadth", type=int, default=2)
    group.add_argument("--gs_iterations", type=int, default=2)

    group = parent_args.add_argument_group("EDPO")
    group.add_argument("--equalizing_preferences", action="store_true", default=False)
    group.add_argument("--max_n_preferences", type=int, default=3)
    group.add_argument("--dpo_beta", type=float, default=0.5)
    group.add_argument("--has_ref_model_constraints", action="store_false", default=True)
    group.add_argument("--edpo_preference_batch_size", type=int, default=2)
    group.add_argument("--ignore_ref_first_n_steps", type=int, default=-1)
    group.add_argument("--save_every_n_episode", type=int, default=1)

    


    return parent_args
     

def get_dataloader_build_func(args, tokenizer:LlamaTokenizer):
    def build_dataloader(replay_buffer, episode, timestep, val_str=""):
        # import datasets as ds
        # gen_data = []
        # reward = []
        # sequence = []
        # action_logprob = []
        # attn_mask = []
        # action_mask = []
        # adv = []
        # values = []
        
        # for item in replay_buffer.items:
        #     # texts,replay_seq, replay_action_pl, replay_action_msk = [],[],[]
        #     texts,replay_seq, replay_action_msk = [],[],[]

        #     for i in range(item.preference_sequences.shape[0]):
        #         seq = item.preference_sequences[i].tolist()
        #         text = tokenizer.decode(seq, skip_special_tokens=False)
        #         texts.append(text.replace('<s>','').replace('</s>', ''))
        #         replay_seq.append(seq)
        #         # replay_action_pl.append(item.action_log_probs[i].tolist())
        #         replay_action_msk.append(item.action_mask[i].tolist())

        #     gen_data.extend(texts)
        #     sequence.extend(replay_seq)
        #     # action_logprob.extend(replay_action_pl)
        #     action_mask.extend(replay_action_msk)
        
        # hf_dataset = ds.Dataset.from_dict({
        #     "item": gen_data,
        #     "sequence": sequence,
        #     "action_mask": action_mask
        # })
        # # "action_logprob": action_logprob,


        rank = mpu.get_data_parallel_rank()
        mp_rank = mpu.get_model_parallel_rank()
        pp_rank = mpu.get_pipe_parallel_rank()
        world_size = mpu.get_data_parallel_world_size()
        global_batch_size = args.policy_train_batch_size * world_size

        # if mp_rank == 0 and pp_rank == 0:
        #     hf_dataset.save_to_disk(os.path.join(args.exp_save_path, f"{args.wandb_project}{val_str}_ep{str(episode).zfill(3)}_{str(timestep).zfill(3)}_{rank}/"))

        # Use a simple sampler with distributed batch sampler
        sampler = torch.utils.data.SequentialSampler(replay_buffer)
        batch_sampler = DistributedBatchSampler(
            sampler=sampler,
            batch_size=global_batch_size,
            drop_last=True,
            rank=rank,
            world_size=world_size,
        )

        # Torch dataloader
        return torch.utils.data.DataLoader(
            replay_buffer,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=replay_buffer.collate_fn,
        )
    return build_dataloader

def get_experience_convert_func(args, src_tokenizer:LlamaTokenizer, dst_tokenizer:LlamaTokenizer):
    
    def convert_func(sequences:torch.Tensor, attention_mask:torch.Tensor, action_mask:torch.Tensor, device:torch.device):

        last_attn_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in attention_mask], dtype=torch.int64)
        
        end_with_eos = torch.diagonal(sequences[:,last_attn_index] == src_tokenizer.eos_token_id)
        
        new_attn_msk = attention_mask.clone().detach()
        new_act_msk = None

        for idx in range(sequences.shape[0]):
            if end_with_eos[idx]:
                new_attn_msk[idx, last_attn_index[idx]] = False
            else:
                text = src_tokenizer.decode(sequences[idx]).replace("\n", "\\n").replace("\t", "\\t")
                print_rank_0(f"TRUNC:{text}")
        
        return sequences, new_attn_msk, new_act_msk
    
    return convert_func
        

def load_dateset(args, tokenizer:LlamaTokenizer):
    
    import datasets as ds
    import random
    from collections import defaultdict
    ds.disable_caching()
    # 'task':[,,...], 'query':[,,...], 'preference':[[],[],...]
    dataset = ds.load_from_disk(args.prompt_dataset_path)
    # [{'input_ids':tensor, 'preference_sequence':tensor, 'task':str}]
    queries = list(dataset["query"])
    if "preference" not in dataset.column_names:
        preferences = []*len(queries)
    else:
        preferences = list(dataset["preference"])
        preferences = [[query+pref for pref in preferences[i]] for i,query in enumerate(queries)]

    samples = []
    for i in range(len(queries)):
        input_ids = tokenizer.encode(queries[i], return_tensors='pt').squeeze(0)
        if len(preferences[i]) > 0:
            preference_ids =  [tokenizer.encode(preference, return_tensors='pt').squeeze(0) for preference in preferences[i]]
            padded_pref_ids = zero_pad_sequences(preference_ids, side="right", padding_value=tokenizer.pad_token_id)
        else:
            padded_pref_ids = torch.tensor([[]])
        samples.append({'input_ids':input_ids, 'preference_sequences':padded_pref_ids})

    if "task" not in dataset.column_names or not args.do_validation:
        return samples, None
    
    tasks = list(dataset["task"])
    task_to_prompts = defaultdict(list)
    for task_name, query in zip(tasks, samples):
        task_to_prompts[task_name].append(query)
    
    train_inputs = []
    val_inputs = []
    for task_name, samples in task_to_prompts.items():
        samples = [{**sample, 'task':task_name} for sample in samples]
        if len(samples) <= args.val_size_per_task:
            train_inputs.extend(samples)
        continue

        g = torch.Generator()                                                                           
        g.manual_seed(args.seed + mpu.get_data_parallel_rank())                                                                       
        random_idx = torch.randperm(len(samples), generator=g).tolist()
        samples = [samples[i] for i in random_idx]

        train_inputs.extend(samples[:-args.val_size_per_task])
        val_inputs.extend(samples[-args.val_size_per_task:])
    
    val_inputs = None if len(val_inputs) == 0 else val_inputs
        
    return train_inputs, val_inputs

def launch(args):
    """Main training program.

    """

    initialize_megatron(args)
    
    strategy = build_deepspeed_config(args)

    llama_tokenizer = AutoTokenizer.from_pretrained(args.policy_tokenizer_path,use_fast=False)
    llama_tokenizer.add_special_tokens(_SPECIAL_TOKENS_DICT)
    llama_tokenizer.add_special_tokens({"additional_special_tokens":[human, bot]})

    rm_model_tokenizer = llama_tokenizer
    # LlamaTokenizer.from_pretrained(args.rm_model_tokenizer_path)


    # print(f"ref model {args.ref_model_path}")
    print(f"ref model {args.policy_model_path}/part_{mpu.get_model_parallel_rank()}")
    # initial model
    # print_rank_0(f'has_ref_model_constraints {args.has_ref_model_constraints}')
    if args.has_ref_model_constraints:
        im = LlamaActor(
            model=LlamaForCausalLM.from_pretrained(
                f"{args.policy_model_path}/part_{mpu.get_model_parallel_rank()}",
                # f"{args.ref_model_path}",
                torch_dtype=torch.bfloat16
            )
        ).eval().to(dtype=torch.bfloat16).cpu() #.to(dtype=torch.bfloat16)
    else:
        im = None
    
    # reward model
    if args.equalizing_preferences:
        reward_model = modeling_fengshenLlama_rm(
            pretrained_path=f"{args.rm_model_path}/part_{mpu.get_model_parallel_rank()}",\
            convert_func=None #get_experience_convert_func(args, src_tokenizer=llama_tokenizer, dst_tokenizer=llama_tokenizer)
        ).eval().half().cpu()
    else:
        reward_model = None


    # deepspeed actor
    print(f"actor model {args.policy_model_path}/part_{mpu.get_model_parallel_rank()}")

    actor_model = LlamaForCausalLM.from_pretrained(
            f"{args.policy_model_path}/part_{mpu.get_model_parallel_rank()}",
            torch_dtype=torch.bfloat16
        )
    if args.activation_checkpointing:                                                                                      
             actor_model.gradient_checkpointing_enable()
    actor = LlamaActor(model=actor_model).to(dtype=torch.bfloat16)
    
 
    actor, actor_optimizer, actor_lr = setup_model_and_optimizer(args, actor, strategy)


    # 初始化experience_maker replay_buffer
    experience_maker = EDPOExperienceMaker(
        actor=actor,
        initial_model=im,
        reward_model=reward_model,
        seed=args.seed + mpu.get_data_parallel_rank(),
        pad_token_id=llama_tokenizer.pad_token_id,
        eos_token_id=llama_tokenizer.eos_token_id,
        gen_minibatch_size=args.gen_minibatch_size,
        actor_minibatch_size=args.policy_minibatch_size,
        rm_minibatch_size=args.rm_minibatch_size,
        gs_args=TotGSArgs(enabling_tot=args.enabling_tot,
                          gs_eval_batch_size=args.gs_eval_batch_size,
                          gs_gen_batch_size=args.gs_gen_batch_size,
                          gs_gen_repeat_times=args.gs_gen_repeat_times,
                          gs_breadth=args.gs_breadth,
                          gs_iterations=args,gs_iterations,
                          generator_tk=llama_tokenizer,
                          evaluator_tk=rm_model_tokenizer),
        equalizing_preferences=args.equalizing_preferences,
        max_n_preferences=args.max_n_preferences)


    replay_buffer = PreferenceReplayBuffer(
        sample_batch_size=args.sample_batch_size,
        limit=args.buffer_limit_size,
        cpu_offload=args.replay_buffer_cpu_offload,
        pad_token_id=llama_tokenizer.pad_token_id
    )

    # 初始化logger
    logger = WandbLogger(
        project=args.wandb_project,
        group=args.wandb_group,
        entity=args.wandb_team,
        ignore=local_rank() != 0,
        name=args.wandb_name,
    )
    logger.log_hyperparams(args)

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
       
    dpo_trainer = EDPOTrainer(
        actor=actor,
        actor_optim=actor_optimizer,
        actor_lr_scheduler = actor_lr,
        experience_maker=experience_maker,
        replay_buffer=replay_buffer,
        beta=args.dpo_beta,
        logger=logger,
        clip_grad=args.clip_grad,
        ckpt_saving_func=get_save_checkpoint_callback(args),
        experience_batch_size=args.experience_batch_size,
        setup_dataloader_func=get_dataloader_build_func(args, llama_tokenizer),
        max_epochs=args.max_epoch_per_update,                  # 每个训练阶段actor和critic的训练轮数
        tokenizer=llama_tokenizer,
        sample_replay_buffer=args.sample_replay_buffer,   # 每次使用全部经验池内容
        **generate_kwargs           # 模型生成样本的参数
    )

    train_inputs, val_inputs = load_dateset(args, llama_tokenizer)
    
    if val_inputs is not None:
        print_rank_0(f"Train on {len(train_inputs)} prompts with {len(val_inputs)} validation every {args.val_check_interval} episode.")
    else:
        print_rank_0(f"Train on {len(train_inputs)} prompts without validation.")

    # 开始训练
    dpo_trainer.fit(
        inputs=train_inputs,
        val_inputs=val_inputs,
        seed=args.seed + mpu.get_data_parallel_rank(),
        val_check_interval=args.val_check_interval,
        save_every_n_episode=args.save_every_n_episode,
        num_episodes=args.num_episodes,
        max_timesteps=args.max_timesteps,     # 每一个episode采样经验的步数
        ignore_ref_first_n_steps=args.ignore_ref_first_n_steps,
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
