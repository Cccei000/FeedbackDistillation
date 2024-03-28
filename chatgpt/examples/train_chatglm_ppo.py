# encoding=utf-8

"""Train"""
import os
import torch
import torch.distributed.run
import deepspeed
import argparse

from fengshen_inner.models.model_utils import (
    add_module_args,
    add_inverse_square_args,
    inverse_square_root_schedule,
    get_scheduler
)
from fengshen_inner.models.megatron import fused_kernels, mpu
from fengshen_inner.strategies.megatron_deepspeed import DeepSpeedStrategy

from chatgpt.dataset import ReplayBufferDataset
from chatgpt.nn.chatglm import ChatGLMCritic, ChatGLMActor
from chatgpt.nn.llama import modeling_llama_rm, LlamaRM
from chatgpt.nn.utils import zero_pad_sequences
from chatgpt.trainer import PPOTrainer
from chatgpt.experience_maker import LocalInferExperienceMaker
from chatgpt.replay_buffer import NaiveReplayBuffer
from chatgpt.logger import WandbLogger
from chatgpt.utils import local_rank

from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from transformers.optimization import AdamW
from transformers import LlamaTokenizer, AutoTokenizer, GenerationConfig

_POLICY_TOKENIZER_PATH = "/cognitive_comp/wanghao/models/chatglm_6b"
_PRETRAIN_MODEL_PATH = '/cognitive_comp/wanghao/models/chatglm_6b'
_REWARD_MODEL_PATH = '/cognitive_comp/liangyuxin/workspace/rm_train/RM_0412_mix_7B/ckpt/last.ckpt/checkpoint/mp_rank_00_model_states.pt'
_REWARD_CONFIG_PATH = '/cognitive_comp/sunqianguo/pretrained/checkpoints/7B/0405/v2/checkpoint-16000/config.json'
_REWARD_TOKENIZER_PATH = "/cognitive_comp/songzhuoyang/models/llama_sft/20230405v1"
_PPO_DATASET_PATH = '/cognitive_comp/songzhuoyang/processed_data/mixed_ppo_dataset_0327_for_chatglm'


def print_rank_0(*message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*message, flush=True)
    else:
        print(*message, flush=True)


def add_neox_ppo_pipeline_args(parent_args:argparse.ArgumentParser):
    
    group = parent_args.add_argument_group("Experiment Args")
    
    group.add_argument("--wandb_project", type=str, default="PPO_ChatGLM")
    group.add_argument("--wandb_group", type=str, default="")
    group.add_argument("--wandb_team", type=str, default=None)
    
    group = parent_args.add_argument_group("Megatron Args")
    group.add_argument("--tensor_model_parallel_size", type=int, default=1)
    group.add_argument("--pipe_model_parallel_size", type=int, default=1)
    group.add_argument("--seed", type=int, default=1234, help="seed")
    group.add_argument("--rank", type=int, default=0)
    group.add_argument("--world_size", type=int, default=64)
    
    group.add_argument("--policy_precision", type=str, default="fp16")
    group.add_argument("--deepspeed_stage", type=int, default=1)
    group.add_argument("--gradient_accumulation_steps", type=int, default=32)
    group.add_argument("--train_micro_batch_size_per_gpu", type=int, default=1)
    group.add_argument("--loss_scale", type=float, default=0)
    group.add_argument("--initial_scale_power", type=int, default=16)
    group.add_argument("--loss_scale_window", type=int, default=1000)
    group.add_argument("--hysteresis", type=int, default=2)
    group.add_argument("--min_loss_scale", type=int, default=2)
     
    group = parent_args.add_argument_group("PPO Args")
    group.add_argument("--num_episodes", type=int, default=1, help="训练轮数，每轮中包括经验池采样、奖励模型打分、模型训练三步")
    group.add_argument("--max_timesteps", type=int, default=1, help="每轮中进行经验池采样的次数")
    group.add_argument("--update_timesteps", type=int, default=1, help="")
    group.add_argument("--sample_replay_buffer", action="store_true", default=False)
    group.add_argument("--sample_batch_size", type=int, default=32, help="每次经验池采样中，使用的Prompt数量（不考虑数据并行）")
    group.add_argument("--buffer_limit_size", type=int, default=512)
    group.add_argument("--max_epoch_per_update", type=int, default=2, help="每次模型训练时，训练的epoch数")
    group.add_argument("--replay_buffer_cpu_offload", type=bool, default=True)
    
    group.add_argument("--eps_clip", type=float, default=0.2)
    group.add_argument("--value_clip", type=float, default=0.2)
    
    group = parent_args.add_argument_group("Experience Args")
    group.add_argument("--top_p", type=float, default=0.95)
    group.add_argument("--top_k", type=int, default=0)
    group.add_argument("--kl_coef", type=float, default=0.0)
    group.add_argument("--max_length", type=int, default=1024)
    group.add_argument("--repetition_penalty", type=float, default=1.)
    group.add_argument("--temperature", type=float, default=1.)
    group.add_argument("--experience_batch_size", type=int, default=32)
    group.add_argument("--policy_minibatch_size", type=int, default=4)
    group.add_argument("--rm_minibatch_size", type=int, default=1)
    group.add_argument("--prompt_dataset_path", type=str, default=_PPO_DATASET_PATH, help="用于训练的所有prompt") # 格式参考默认路径的dataset
    group.add_argument("--exp_save_path", type=str, default="/cognitive_comp/songzhuoyang/workspace/chatgpt/chatglm_rlhf/exp", help="训练产生的经验池的保存路径") 
    
    group = parent_args.add_argument_group("Trainer Args")
    group.add_argument("--num_workers", type=int, default=2)
    group.add_argument("--total_steps", type=int, default=1e4)
    group.add_argument("--policy_train_batch_size", type=int, default=1)
    
    group = parent_args.add_argument_group("Model Args")
    group.add_argument("--policy_ckpt_path", type=str, default="/cognitive_comp/songzhuoyang/workspace/chatgpt/chatglm_rlhf/ckpt", help="rlhf ckpt保存的根目录") # 训练过程中会自动创建文件夹，保存每个episode的ckpt
    group.add_argument("--policy_tokenizer_path", type=str, default=_POLICY_TOKENIZER_PATH) # tokenizer路径
    group.add_argument("--policy_model_path", type=str, default=_PRETRAIN_MODEL_PATH, help="生成模型sft ckpt路径") # 需要使用gpt-neox/utils中的gxy写的转换脚本转换参数的key
    group.add_argument("--rm_config_path", type=str, default=_REWARD_CONFIG_PATH, help="rm config路径") # 训练过程中会自动创建文件夹，保存每个episode的ckpt
    group.add_argument("--rm_model_path", type=str, default=_REWARD_MODEL_PATH, help="rm ckpt路径") # 需要使用gpt-neox/utils中的gxy写的转换脚本转换参数的key_REWARD_TOKENIZER_PATH
    group.add_argument("--rm_model_tokenizer_path", type=str, default=_REWARD_TOKENIZER_PATH) # rm tokenizer路径
    group.add_argument("--rm_model_max_seq_len", type=int, default=1024) # rm tokenizer路径
    
    return parent_args
     

def initialize_megatron(tensor_model_parallel_size, pipe_model_parallel_size, seed):

    fused_kernels.load_fused_kernels()

    deepspeed.init_distributed(
        dist_backend='nccl',
        distributed_port=os.getenv("MASTER_PORT", "6000"),
        verbose=False,
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    device_count = torch.cuda.device_count()
    device = rank % device_count
    torch.cuda.set_device(device)
    
    print(f"Device count: {device_count} | Rank: {rank}")

    from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology

    # this does pipe on the most outside, then data, then model.
    # PipeModelDataParallelTopology is just a wrapper over ProcessTopology that predefines this order.
    dp = world_size // pipe_model_parallel_size // tensor_model_parallel_size
    topo = PipeModelDataParallelTopology(num_pp=pipe_model_parallel_size,
                                         num_mp=tensor_model_parallel_size,
                                         num_dp=dp)

    # Offset base seeds for the interior pipeline stages.
    # TODO: adjust last stage too once IO is improved.
    stage_id = topo.get_coord(rank=rank).pipe
    if 0 < stage_id < topo.get_dim("pipe") - 1:
        offset = seed + 1138
        seed = offset + (stage_id * tensor_model_parallel_size)

    mpu.initialize_model_parallel(
        tensor_model_parallel_size,
        topology=topo,
        fp32_allreduce=False)

    deepspeed.checkpointing.configure(
        mpu, partition_activations=True)

    mpu.model_parallel_cuda_manual_seed(seed)
    mpu.set_model_parallel_world_size(tensor_model_parallel_size)
    # and return function for external DDP manager to call when it has DDP initialized
    mpu.set_model_parallel_rank(rank)


def get_default_update_params(pl_model: torch.nn.Module, weight_decay:float):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'layer_norm.', 'layernorm.']
    optimizer_grouped_params = [
        {'params': [p for n, p in pl_model.named_parameters() if not any(
            nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': weight_decay},
        {'params': [p for n, p in pl_model.named_parameters() if any(
            nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]
    return optimizer_grouped_params


def configure_optimizers(args, model, strategy=None):
    '''
    Args:
    '''
    # get params that optimizer need
    optimizer_grouped_params = get_default_update_params(model, args.weight_decay)

    # Configure optimizer.
    if strategy is not None:
        if 'offload_optimizer' in strategy.config['zero_optimization']:
            optimizer = DeepSpeedCPUAdam(
                optimizer_grouped_params, adamw_mode=True,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon)
        else:
            optimizer = FusedAdam(
                optimizer_grouped_params, adam_w_mode=True,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon)
    else:
        optimizer = AdamW(optimizer_grouped_params, lr=args.learning_rate,
                          betas=(args.adam_beta1, args.adam_beta2),
                          eps=args.adam_epsilon)
    # Configure learning rate scheduler.
    total_steps = args.lr_decay_ratio * \
        args.total_steps if args.lr_decay_steps == 0 else args.lr_decay_steps
    warmup_steps = args.warmup_ratio * \
        args.total_steps if args.warmup_steps == 0 else args.warmup_steps

    if args.scheduler_type == "inverse_sqrt":
        scheduler = inverse_square_root_schedule(optimizer=optimizer,
                                                 num_warmup_steps=warmup_steps, lr_min=args.warmup_min_lr, lr_max=args.warmup_max_lr)
    else:
        scheduler = get_scheduler(name=args.scheduler_type, optimizer=optimizer,
                                  num_warmup_steps=warmup_steps, num_training_steps=total_steps,
                                  lr_end=args.min_learning_rate)
    # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    return optimizer, scheduler, optimizer_grouped_params


def get_total_params(model):
    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        params = sum([p.nelement() for p in model.parameters()])
        print(
            " > number of parameters on model parallel rank {}: {}".format(
                mpu.get_model_parallel_rank(), params
            ),
            flush=True,
        )
    else:
        params = 0

    total_n_parameters = torch.tensor([params]).cuda(torch.cuda.current_device())
    torch.distributed.all_reduce(total_n_parameters)
    total_n_parameters = total_n_parameters.item()
    return total_n_parameters


def setup_model_and_optimizer(args, model, strategy):
    '''
    返回model, optimizer, lr_scheduler
    '''

    optimizer, lr_scheduler, model_params = configure_optimizers(args, model, strategy)
    
    print_rank_0("DeepSpeed is enabled.")
    rank = torch.distributed.get_rank()
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=argparse.Namespace(device_rank=rank),
        model=model,
        optimizer=optimizer,
        config=strategy.config,
        lr_scheduler=lr_scheduler,
        dist_init_required=False,
        model_parameters=model_params,
        mpu=mpu,
    )
    model.total_params = get_total_params(model.module)
    print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')

    return model, optimizer, lr_scheduler


def get_dataloader_build_func(args, tokenizer:AutoTokenizer):
    def build_dataloader(replay_buffer, episode, timestep):
        
        dataset = ReplayBufferDataset(replay_buffer, pad_token_id=tokenizer.pad_token_id)
        
        import datasets as ds
        gen_data = []
        reward = []
        sequence = []
        action_logprob = []
        attn_mask = []
        action_mask = []
        adv = []
        
        for item in dataset:
            text = tokenizer.decode(item.sequences, skip_special_tokens=True)
            gen_data.append(text)
            reward.append(item.reward.item())
            sequence.append(item.sequences.tolist())
            action_logprob.append(item.action_log_probs.tolist())
            attn_mask.append(item.attention_mask.tolist())
            action_mask.append(item.action_mask.tolist())
            adv.append(item.advantages.item())
        
        hf_dataset = ds.Dataset.from_dict({
            "item": gen_data,
            "reward": reward,
            "advantage": adv,
            "sequence": sequence,
            "action_logprob": action_logprob,
            "attn_mask": attn_mask,
            "action_mask": action_mask
        })
        hf_dataset.save_to_disk(os.path.join(args.exp_save_path, f"{args.wandb_project}_ep{str(episode).zfill(3)}_{str(timestep).zfill(3)}_{torch.distributed.get_rank()}/"))

        # print(f"device:{example.sequences.device} - {example.reward.item()} - {text}")
        print_rank_0(f"adv_mean:{dataset.collate_fn.adv_mean} | adv_std:{1 / dataset.collate_fn.inverse_adv_std}")
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            batch_size=args.policy_train_batch_size,
        )
    return build_dataloader


def get_experience_convert_func(args, src_tokenizer:AutoTokenizer, dst_tokenizer:LlamaTokenizer):
    
    import re
    
    def process_text(text:str) -> str:
        
        text = text.replace("\n答:", "\n\nAssistant:")
        text = re.sub(r"^\[Round 0\]\n问:", "Human:", text)
        text = re.sub(r"\[Round [\d]+\]\n问：", "\n\nHuman:", text)
        
        return text
    
    def convert_func(sequences:torch.Tensor, attention_mask:torch.Tensor, action_mask:torch.Tensor, device:torch.device):
        
        try:
            sequence_texts = src_tokenizer.batch_decode(sequences) # 完整的句子
        except Exception as e:
            print(sequences.tolist())
            raise e
        
        quert_ids = sequences.clone().detach()
        quert_ids[:, -action_mask.shape[1]:] = torch.where(
            action_mask==0,
            quert_ids[:, -action_mask.shape[1]:],
            torch.full_like(quert_ids[:, -action_mask.shape[1]:], src_tokenizer.pad_token_id, device=quert_ids.device)
        )
        query_texts = src_tokenizer.batch_decode(quert_ids) # 仅query
        
        token_ids = []
        attn_msk = []
        act_msk = []
        
        for seq_text, query_text in zip(sequence_texts, query_texts):
            seq_text = process_text(seq_text)
            # seq_text += dst_tokenizer.eos_token
            query_text = process_text(query_text)
           
            seq_encode_dict = dst_tokenizer(seq_text)
            query_encode_dict = dst_tokenizer(query_text)
            
            seq_len = min(len(seq_encode_dict["input_ids"]), args.rm_model_max_seq_len)
            act_length = seq_len - len(query_encode_dict["input_ids"])
            if len(seq_encode_dict["input_ids"]) > args.rm_model_max_seq_len:
                print(f"Truncate respond: {len(seq_encode_dict['input_ids'])} -> {args.rm_model_max_seq_len}")
            
            token_ids.append(torch.tensor(seq_encode_dict["input_ids"])[:seq_len])
            attn_msk.append(torch.tensor(seq_encode_dict["attention_mask"])[:seq_len])
            act_msk.append(torch.ones(act_length))

        # print_rank_0(f'sequence_texts {sequence_texts} query_texts {query_texts}')
        token_ids = zero_pad_sequences(token_ids, "left", padding_value=dst_tokenizer.pad_token_id).to(device=device)
        attn_msk = zero_pad_sequences(attn_msk, "left", padding_value=0).to(device=device)
        act_msk = zero_pad_sequences(act_msk, "left", padding_value=0).to(device=device)
        
        return token_ids, attn_msk, act_msk
    
    return convert_func
            

def get_save_checkpoint_callback(args):
    
    def saving(iteration, model:deepspeed.DeepSpeedEngine, **kwargs):
        
        tag = f"global_step{iteration}"
        model.save_checkpoint(args.policy_ckpt_path, tag=tag)

    return saving


def build_deepspeed_config(args):
    
    # 使用DeepSpeedStrategy读取deepspeed配置
    strategy = DeepSpeedStrategy(
        zero_optimization=True,
        stage=args.deepspeed_stage,
        partition_activations=True,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipe_model_parallel_size=args.pipe_model_parallel_size,
        mpu_seed=args.seed,
    )
    strategy.config.setdefault("train_micro_batch_size_per_gpu", args.train_micro_batch_size_per_gpu)
    strategy.config["gradient_accumulation_steps"]  = args.gradient_accumulation_steps

    if args.policy_precision == "fp16":
        print_rank_0("Enabling DeepSpeed FP16.")
        strategy.config["fp16"] = {
            "enabled": True,
            "loss_scale": args.loss_scale,
            "initial_scale_power": args.initial_scale_power,
            "loss_scale_window": args.loss_scale_window,
            "hysteresis": args.hysteresis,
            "min_loss_scale": args.min_loss_scale,
        }
    elif args.policy_precision == "bf16":
        print_rank_0("Enabling DeepSpeed BF16.")
        strategy.config["fp16"] = {"enabled": True}
        
    return strategy


def launch(args):
    """Main training program.

    """

    initialize_megatron(
        args.tensor_model_parallel_size,
        args.pipe_model_parallel_size,
        args.seed
    )
    
    strategy = build_deepspeed_config(args)
    
    chatglm_tokenizer = AutoTokenizer.from_pretrained(args.policy_tokenizer_path, trust_remote_code=True)
    llama_tokenizer = LlamaTokenizer.from_pretrained(args.rm_model_tokenizer_path)

    # initial model
    im = ChatGLMActor(
        ckpt_path=args.policy_model_path, generation_config=None, bos_token_id=chatglm_tokenizer.bos_token_id, pad_token_id=chatglm_tokenizer.pad_token_id, gmask_token_id=chatglm_tokenizer.gmask_token_id
    ).half().eval().cpu()
    
    # reward model
    # reward_model = GPTNeoXRM(GPTNeoXModel.from_pretrained(_REWARD_MODEL_PATH)).half().cpu()
    reward_model = modeling_llama_rm(
        config_path=args.rm_config_path, ckpt_path=args.rm_model_path, convert_func=get_experience_convert_func(args, src_tokenizer=chatglm_tokenizer, dst_tokenizer=llama_tokenizer)
    ).eval().half().cpu()
    
    generate_kwargs = {
        "do_sample": True,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_length": args.max_length,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "pad_token_id": chatglm_tokenizer.pad_token_id,
        "eos_token_id": chatglm_tokenizer.eos_token_id,
    }
    
    tokenizer_vocab_size = chatglm_tokenizer.vocab_size
    policy_vocab_size = im.model.config.vocab_size
    
    if policy_vocab_size > tokenizer_vocab_size:
        bad_words_ids = [[ids] for ids in range(tokenizer_vocab_size, policy_vocab_size)]
        print_rank_0(f"BAD TOKEN IDS: {tokenizer_vocab_size}~{policy_vocab_size - 1}")
    else:
        bad_words_ids = None

    # deepspeed actor & critic
    actor = ChatGLMActor(
        ckpt_path=args.policy_model_path,
        generation_config=GenerationConfig(
            num_return_sequences=1,
            bad_words_ids=bad_words_ids,
            **generate_kwargs
        ), bos_token_id=chatglm_tokenizer.bos_token_id, pad_token_id=chatglm_tokenizer.pad_token_id, gmask_token_id=chatglm_tokenizer.gmask_token_id
    ).half()#.to(torch.bfloat16)
    critic = ChatGLMCritic(ckpt_path=args.policy_model_path, bos_token_id=chatglm_tokenizer.bos_token_id, pad_token_id=chatglm_tokenizer.pad_token_id, gmask_token_id=chatglm_tokenizer.gmask_token_id).half()#.to(torch.bfloat16)
    actor, actor_optimizer, actor_lr = setup_model_and_optimizer(args, actor, strategy)
    critic, critic_optimizer, critic_lr = setup_model_and_optimizer(args, critic, strategy)

    # 初始化experience_maker replay_buffer

    experience_maker = LocalInferExperienceMaker(
        actor=actor, critic=critic, reward_model=reward_model, initial_model=im, kl_coef=args.kl_coef, pad_token_id=chatglm_tokenizer.pad_token_id,
        actor_minibatch_size=args.policy_minibatch_size,
        rm_minibatch_size=args.rm_minibatch_size
    )

    replay_buffer = NaiveReplayBuffer(
        sample_batch_size=args.sample_batch_size,
        limit=args.buffer_limit_size,
        cpu_offload=args.replay_buffer_cpu_offload,
        pad_token_id=chatglm_tokenizer.pad_token_id,
    )

    # 初始化logger
    logger = WandbLogger(
        project=args.wandb_project,
        group=args.wandb_group,
        entity=args.wandb_team,
        ignore=local_rank() != 0,
    )
    logger.log_hyperparams(args)

    # 初始化trainer

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
        ckpt_saving_func=get_save_checkpoint_callback(args),
        experience_batch_size=args.experience_batch_size,
        setup_dataloader_func=get_dataloader_build_func(args, chatglm_tokenizer),
        eps_clip=args.eps_clip,                                # ratio 裁剪
        value_clip=args.value_clip,                            # value 裁剪
        max_epochs=args.max_epoch_per_update,                  # 每个训练阶段actor和critic的训练轮数
        tokenizer=chatglm_tokenizer,
        sample_replay_buffer=args.sample_replay_buffer,   # 每次使用全部经验池内容
        **generate_kwargs           # 模型生成样本的参数
    )

    import datasets as ds
    ds.disable_caching()
    dataset = ds.load_from_disk(args.prompt_dataset_path)
    prompts = list(dataset["query"])

    # 开始训练
    ppo_trainer.fit(
        prompts=prompts,
        num_episodes=args.num_episodes,
        max_timesteps=args.max_timesteps,     # 每一个episode采样经验的步数
        update_timesteps=args.update_timesteps,   # 训练模型的累积步数
    )
    return


if __name__ == "__main__":
    # 复用deepy.py的代码，在这里生成所有的参数，前三个参数是不需要的，直接丢掉
    # neox_args = NeoXArgs.consume_deepy_args()
    # deepspeed_main_args = neox_args.get_deepspeed_main_args()
    # args = deepspeed_main_args[2:]

    # 传入我们生成的参数列表
    # neox_args = NeoXArgs.consume_neox_args(args=args)
    # neox_args.configure_distributed_args()
    # tokenizer needs to be build in training in order to set the padding vocab
    # neox_args.build_tokenizer()
    # is initialized if tensorboard directory is defined
    # neox_args.initialize_tensorboard_writer()

    # set_args(neox_args)
    # print(neox_args)
    # pretrain(neox_args=neox_args)
    
    parser = argparse.ArgumentParser()
    parser = add_module_args(parent_args=parser)
    parser = add_inverse_square_args(parent_args=parser)
    parser = add_neox_ppo_pipeline_args(parent_args=parser)
    
    args = parser.parse_args()
    
    launch(args)
    print("!")
