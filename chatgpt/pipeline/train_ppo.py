# encoding=utf-8
"""Launch PPO Pipeline"""
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.run
from deepspeed.profiling.flops_profiler import FlopsProfiler
from fengshen_inner.models.llama.configuration_llama import LlamaConfig
from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM
from fengshen_inner.models.llama.modeling_llama_lora import (
    LlamaForCausalLMLora, LlamaModelLora)
from fengshen_inner.models.megatron import mpu
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer

from chatgpt.dataset import ReplayBufferDataset
from chatgpt.experience_maker import (PPOPPExperienceMaker,
                                      StepLevelExperienceMaker)
from chatgpt.logger import WandbLogger
from chatgpt.nn.llama import (LlamaActor, LlamaCritic,
                              modeling_fengshenLlama_critic,
                              modeling_fengshenLlama_rm)
from chatgpt.pipeline.config import (ActorGranularity, PPOPipelineConfig,
                                     RewardModelGranularity)
from chatgpt.replay_buffer import DistributedBatchSampler, NaiveReplayBuffer
from chatgpt.strategies import (build_deepspeed_config,
                                get_save_checkpoint_callback,
                                initialize_megatron, setup_inference_engine,
                                setup_model_and_optimizer)
from chatgpt.trainer import PPOPPTrainer, PPOTrainer
from chatgpt.utils import (ACTOR_INFER, ACTOR_TRAIN, CRITIC_INFER,
                           CRITIC_TRAIN, GENERATE, REF, RM,
                           DeepspeedFlopsTimerGroup, FlopsTimer,
                           FlopsTimerGroup, LoggingLevel, local_rank,
                           logging_rank_0)
from chatgpt.nn import GSArgs

from .tokenizer import GLUE, TOKENIZER
from .utils import concat_prompt, load_jsonline_data, save_dataset_to_jsonl


def get_dataloader_build_func(args:PPOPipelineConfig, tokenizer:LlamaTokenizer, exp_save_path:str, ppo_granularity:ActorGranularity) -> callable:
    """根据args生成dataloader构造方法

    Args:
        args (PPOPipelineConfig):           PPO流程配置
        tokenizer (LlamaTokenizer):         tokenizer
        exp_save_path (str):                经验池保存路径
        ppo_granularity (PPOGranularity):   PPO训练流程采用的粒度

    Returns:
        callable: dataloader构造方法
    """    
    def build_dataloader(replay_buffer:NaiveReplayBuffer, episode:int, timestep:int) -> DataLoader:
        """dataloader构造方法。
        本方法传入Trainer中。在每次开始训练之前，调用本方法基于经验池构造dataloader。

        Args:
            replay_buffer (NaiveReplayBuffer): 经验池
            episode (int): episode
            timestep (int): episode内的time step

        Returns:
            DataLoader: dataloader
        """        
        
        # step_level_ppo:处理训练集samples数一致
        if ppo_granularity is ActorGranularity.step:
            sample_count = torch.tensor([len(replay_buffer)], device=torch.cuda.current_device())
            dist.all_reduce(sample_count, dist.ReduceOp.MIN)
            replay_buffer.items = replay_buffer.items[:sample_count]
        
        dataset = ReplayBufferDataset(
            replay_buffer, pad_token_id=tokenizer.pad_token_id, gamma=args.gamma, lam=args.lam, ppo_granularity=ppo_granularity
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
            text = text.replace('<s>','').replace('</s>', '').replace('<unk>', '').strip()

            gen_data.append(text)
            sequence.append(item.sequences.tolist())
            action_logprob.append(item.action_log_probs.tolist())
            attn_mask.append(item.attention_mask.tolist())
            action_mask.append(item.action_mask.tolist())
            # Token-level PPO
            if ppo_granularity is ActorGranularity.token:
                adv.append(item.advantages.tolist())
                reward.append(item.origin_reward.tolist())
                ret.append(item.reward.tolist())
                values.append(item.values.tolist())
            # Sample-level PPO || Step-level PPO
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

        # 保存本次用于训练的经验池样本
        if mp_rank == 0 and pp_rank == 0:
            
            save_dataset_to_jsonl(hf_dataset, path=os.path.join(exp_save_path, f"{args.wandb_project}_ep{str(episode).zfill(3)}_{str(timestep).zfill(3)}_{rank}.jsonl"))
            hf_dataset.save_to_disk(os.path.join(exp_save_path, f"{args.wandb_project}_ep{str(episode).zfill(3)}_{str(timestep).zfill(3)}_{rank}/"))
            logging_rank_0(f"adv_mean:{dataset.collate_fn.adv_mean} | adv_std:{1 / dataset.collate_fn.inverse_adv_std}", "debug")

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


def get_val_experience_saving_func(args:PPOPipelineConfig, tokenizer:LlamaTokenizer, exp_save_path:str, ppo_granularity: ActorGranularity) -> callable:
    """根据args生成验证集保存方法

    Args:
        args (PPOPipelineConfig):           PPO流程配置
        tokenizer (LlamaTokenizer):         tokenizer
        exp_save_path (str):                经验池保存路径
        ppo_granularity (PPOGranularity):   PPO训练流程采用的粒度

    Returns:
        callable: 验证集保存方法
    """    
    def save_val_experience(replay_buffer:NaiveReplayBuffer, episode:int, task_name:List[str]) -> None:
        """验证集保存方法
        本方法传入Trainer中。在每次验证集推理完成后，调用本方法进行保存。

        Args:
            replay_buffer (NaiveReplayBuffer):  验证集推理结果
            episode (int):                      episode
            task_name (List[str]):              每个样本对应的任务类型
        """        
        dataset = ReplayBufferDataset(replay_buffer, pad_token_id=tokenizer.pad_token_id, gamma=args.gamma, lam=args.lam, ppo_granularity=ppo_granularity, for_validation=True)
        
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
            text = text.replace('<s>','').replace('</s>', '').replace('<unk>', '').strip()
            gen_data.append(text)
            sequence.append(item.sequences.tolist())
            action_logprob.append(item.action_log_probs.tolist())
            attn_mask.append(item.attention_mask.tolist())
            action_mask.append(item.action_mask.tolist())
            # Token-level PPO
            if ppo_granularity is ActorGranularity.token:
                adv.append(item.advantages.tolist())
                reward.append(item.origin_reward.tolist())
                ret.append(item.reward.tolist())
                values.append(item.values.tolist())
            # Sample-level PPO || Step-level PPO
            else:
                adv.append(item.advantages.item())
                reward.append(item.origin_reward.item())
                ret.append(item.reward.item())
                values.append(item.values.tolist())
            
        assert len(task_name) == len(gen_data), f"TASK:{len(task_name)} != DATA:{len(gen_data)}"
        
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
            save_dataset_to_jsonl(hf_dataset, path=os.path.join(exp_save_path, f"{args.wandb_project}_val_ep{str(episode).zfill(3)}_{rank}.jsonl"))
            hf_dataset.save_to_disk(os.path.join(exp_save_path, f"{args.wandb_project}_val_ep{str(episode).zfill(3)}_{rank}/"))
        return
    return save_val_experience
        

def load_dateset(args:PPOPipelineConfig) -> Tuple[Dict[str,List[str]], Dict[str,List[str]], Dict[str,List[str]], Dict[str,str]]:
    """加载jsonl格式的数据，根据是否存在guidance response分离为free task和guide task，同时根据配置划分训练集和验证集
    输出按照{任务类型：问题列表}来组织

    Args:
        args (PPOPipelineConfig): PPO流程配置

    Returns:
        Tuple[Dict[str,List[str]], Dict[str,List[str]], Dict[str,List[str]], Dict[str,str]]: （无引导的训练集样本，有引导的训练及样本，验证集样本，各样本的引导回复）
    """    
    
    from collections import defaultdict
    
    dataset = load_jsonline_data(
        path=args.dataset_path,
        prefix=args.prefix,
        seperator=args.multiturn_seperator
    )
    dataset = concat_prompt(
        dataset=dataset, prefix=args.prefix, seperator=args.multiturn_seperator
    )
    
    queries = list(dataset["query"])
    guides = list(dataset["golden_res"]) if "golden_res" in dataset.column_names else [""] * dataset.shape[0]
    tasks = list(dataset["task"])
    
    free_task_to_prompts = defaultdict(list)
    guide_task_to_prompts = defaultdict(list)
    prompt_to_guidance = defaultdict(str)
    
    # 分离引导任务
    for task_name, query, guide in zip(tasks, queries, guides):
        if len(guide) > 0:
            prompt_to_guidance[query] = guide
            guide_task_to_prompts[task_name].append(query)
        else:
            free_task_to_prompts[task_name].append(query)
    
    guide_task_names = list(guide_task_to_prompts.keys())
    
    # 清理free task中的guide task
    for task_name in guide_task_names:
        if task_name in free_task_to_prompts.keys():
            del free_task_to_prompts[task_name]
    
    # 分离验证集
    val_queries = {}
    for task_name, queries in free_task_to_prompts.items():
        g = torch.Generator()                                                                           
        g.manual_seed(args.seed + mpu.get_data_parallel_rank())                                                                       
        random_idx = torch.randperm(len(queries), generator=g).tolist()
        queries = [queries[i] for i in random_idx]
        val_queries[task_name] = queries[-args.val_size_per_task:]
        free_task_to_prompts[task_name] = queries[:-args.val_size_per_task]
    
    for task_name, queries in guide_task_to_prompts.items():
        g = torch.Generator()                                                                           
        g.manual_seed(args.seed + mpu.get_data_parallel_rank())                                                                       
        random_idx = torch.randperm(len(queries), generator=g).tolist()
        queries = [queries[i] for i in random_idx]
        
        val_queries[task_name] = queries[-args.val_size_per_task:]
        guide_task_to_prompts[task_name] = queries[:-args.val_size_per_task]
    
    return free_task_to_prompts, guide_task_to_prompts, val_queries, prompt_to_guidance


def launch_ppo(args:PPOPipelineConfig, rm_ckpt_path:Optional[str]=None):
    """启动PPO训练流程

    Args:
        args (PPOPipelineConfig): PPO流程配置
        rm_ckpt_path (Optional[str], optional): 暂未使用，留给后续连续训练RM和PPO时使用. Defaults to None.
    """    
    
    initialize_megatron(args)
    # logging_initialize(level=args.logging_level)
    strategy = build_deepspeed_config(args)
    
    ### 准备用于保存的目录
    # root
    workspace_path = args.workspace_path
    logging_path = args.logging_path
    os.makedirs(workspace_path, exist_ok=True)
    # exp
    exp_save_path = os.path.join(workspace_path, f"exp")
    os.makedirs(exp_save_path, exist_ok=True)
    # ckpt
    ckpt_save_path = os.path.join(workspace_path, f"ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    # tensorboard logs
    tb_log_path = os.path.join(logging_path, f"runs")
    os.makedirs(tb_log_path, exist_ok=True)
    
    # 初始化logger
    logger = WandbLogger(
        project=args.wandb_project,
        group=args.wandb_group,
        entity=args.wandb_team,
        ignore=local_rank() != 0,
        name=f"ppo-{args.wandb_name}",
        tensorboard_dir=tb_log_path
    )
    logger.log_hyperparams(args.__dict__)
    

    policy_tokenizer = TOKENIZER[args.policy_model_type](tokenizer_path=args.policy_tokenizer_path)
    rm_tokenizer = TOKENIZER[args.reward_model_type](tokenizer_path=args.rm_tokenizer_path)
    
    ### 模型加载路径
    policy_model_path = os.path.join(workspace_path, f"models/policy")
    rm_model_path = os.path.join(workspace_path, "models/reward_model")

    ### 加载模型
    policy_precision = torch.bfloat16 if args.policy_precision == "bf16" else torch.float16
    rm_precision = torch.bfloat16 if args.rm_precision == "bf16" else torch.float16
    ppo_granularity = ActorGranularity[args.actor_granularity]
    rm_granularity = RewardModelGranularity[args.rm_granularity]
    
    logging_rank_0(f"RM Granularity is '{rm_granularity.value}'")
    logging_rank_0(f"PPO Granularity is '{ppo_granularity.value}'")
    
    im_pure = LlamaForCausalLM.from_pretrained(
        os.path.join(policy_model_path, f"part_{mpu.get_model_parallel_rank()}"),
        torch_dtype=policy_precision
    )
    
    #### Enable lora ####
    if args.enable_policy_lora:
        logging_rank_0(f"Enable Policy LoRA with lora_rank={args.lora_rank}.")
        actor_config = deepcopy(im_pure.config)
        actor_config.torch_dtype =policy_precision
        actor_config.lora = True
        actor_config.lora_alpha = args.lora_alpha
        actor_config.lora_r = args.lora_rank
        actor_config.lora_dropout = args.lora_dropout
        
        actor_pure = LlamaForCausalLMLora(actor_config, im_pure).to(dtype=policy_precision)
        critic_pure = LlamaModelLora(actor_config, im_pure.llama).to(dtype=policy_precision)
        actor_pure.enable_lora()
        critic_pure.enable_lora()
        
        actor = LlamaActor(
            model=actor_pure,
            actor_granularity=ppo_granularity,
            rm_granularity=rm_granularity,
        ).to(dtype=policy_precision)
        critic = LlamaCritic(
            model=critic_pure,
            ppo_granularity=ppo_granularity,
        ).to(dtype=policy_precision)
    else:
        actor = LlamaActor(
            model=LlamaForCausalLM.from_pretrained(
                os.path.join(policy_model_path, f"part_{mpu.get_model_parallel_rank()}"),
                torch_dtype=policy_precision
            ),
            actor_granularity=ppo_granularity,
            rm_granularity=rm_granularity,
        ).to(dtype=policy_precision)
        critic = modeling_fengshenLlama_critic(
            pretrained_path=os.path.join(policy_model_path if args.critic_from_sft else rm_model_path, f"part_{mpu.get_model_parallel_rank()}"),
            ppo_granularity=ppo_granularity
        ).to(dtype=policy_precision)
    
    
    # initial model
    im = LlamaActor(
        model=im_pure,
        actor_granularity=ppo_granularity,
        rm_granularity=rm_granularity,
    ).eval().to(dtype=policy_precision).cpu()
    
    #### 对齐词表 ####
    tokenizer_vocab_size = policy_tokenizer.vocab_size
    policy_vocab_size = im.model.config.vocab_size

    bad_words_ids = [[policy_tokenizer.convert_tokens_to_ids(role)] for role in ['<human>','<bot>']]
    logging_rank_0(f"Ignore all role tokens: {bad_words_ids}", LoggingLevel.DEBUG)

    if policy_vocab_size > tokenizer_vocab_size:
        bad_words_ids += [[ids] for ids in range(tokenizer_vocab_size, policy_vocab_size)]
        logging_rank_0(f"BAD TOKEN IDS: {tokenizer_vocab_size}~{policy_vocab_size - 1}", LoggingLevel.DEBUG)
    ####
    if not args.enable_constrain_actor:
        im = setup_inference_engine(model=im, mp_size=args.tensor_model_parallel_size, dtype=policy_precision)

    # reward model
    model_glue = GLUE.get(f"{args.policy_model_type}_to_{args.reward_model_type}", None)
    glue = None if model_glue is None else model_glue(args=args, src_tokenizer=policy_tokenizer, dst_tokenizer=rm_tokenizer)
    
    if glue is None:
        logging_rank_0(f"Same model type, not using prompt glue.", "debug")
    else:
        logging_rank_0(f"Different model type ({args.policy_model_type} policy & {args.reward_model_type} RM), using prompt glue.", "debug")
    
    reward_model = modeling_fengshenLlama_rm(
        pretrained_path=os.path.join(rm_model_path, f"part_{mpu.get_model_parallel_rank()}"),
        convert_func=glue,
        actor_granularity=ppo_granularity,
        rm_granularity=rm_granularity,
        logger=logger,
    ).eval().to(dtype=rm_precision).cpu()

    reward_model = setup_inference_engine(model=reward_model, mp_size=args.tensor_model_parallel_size, dtype=rm_precision)

    
    logging_rank_0(f"Actor lr scheduler type is '{args.actor_scheduler_type}'")
    logging_rank_0(f"Critic lr scheduler type is '{args.critic_scheduler_type}'")
    
    actor, actor_optimizer, actor_lr = setup_model_and_optimizer(args, actor, strategy, learning_rate=args.actor_lr, scheduler_type=args.actor_scheduler_type)
    critic, critic_optimizer, critic_lr = setup_model_and_optimizer(args, critic, strategy, learning_rate=args.critic_lr, scheduler_type=args.critic_scheduler_type)
    
    policy_config = LlamaConfig.from_pretrained(policy_model_path)  
    
    logging_rank_0(f"Policy: hidden_size={policy_config.hidden_size}, num_layers={policy_config.num_hidden_layers}, total_params={actor.total_params}", LoggingLevel.DEBUG)
    logging_rank_0(f"RM: hidden_size={policy_config.hidden_size}, num_layers={policy_config.num_hidden_layers}, total_params={critic.total_params}", LoggingLevel.DEBUG)
    
    flop_timers = FlopsTimerGroup(timers={
        GENERATE: FlopsTimer(
            hidden_size=policy_config.hidden_size,
            num_layers=policy_config.num_hidden_layers,
            total_params=actor.total_params,
            vocab_size=policy_config.vocab_size,
            world_size=mpu.get_model_parallel_world_size(),
            is_train=False,
        ),
        ACTOR_TRAIN: FlopsTimer(
            hidden_size=policy_config.hidden_size,
            num_layers=policy_config.num_hidden_layers,
            total_params=actor.total_params,
            vocab_size=policy_config.vocab_size,
            world_size=mpu.get_model_parallel_world_size(),
            is_train=True,
        ),
        ACTOR_INFER: FlopsTimer(
            hidden_size=policy_config.hidden_size,
            num_layers=policy_config.num_hidden_layers,
            total_params=actor.total_params,
            vocab_size=policy_config.vocab_size,
            world_size=mpu.get_model_parallel_world_size(),
            is_train=False,
        ),
        REF: FlopsTimer(
            hidden_size=policy_config.hidden_size,
            num_layers=policy_config.num_hidden_layers,
            total_params=actor.total_params,
            vocab_size=policy_config.vocab_size,
            world_size=mpu.get_model_parallel_world_size(),
            is_train=False,
        ),
        # FIXME: 读取RM的配置
        CRITIC_TRAIN: FlopsTimer(
            hidden_size=policy_config.hidden_size,
            num_layers=policy_config.num_hidden_layers,
            total_params=critic.total_params,
            vocab_size=policy_config.vocab_size,
            world_size=mpu.get_model_parallel_world_size(),
            is_train=True,
        ),
        CRITIC_INFER: FlopsTimer(
            hidden_size=policy_config.hidden_size,
            num_layers=policy_config.num_hidden_layers,
            total_params=critic.total_params,
            vocab_size=policy_config.vocab_size,
            world_size=mpu.get_model_parallel_world_size(),
            is_train=False,
        ),
        RM: FlopsTimer(
            hidden_size=policy_config.hidden_size,
            num_layers=policy_config.num_hidden_layers,
            total_params=critic.total_params,
            vocab_size=policy_config.vocab_size,
            world_size=mpu.get_model_parallel_world_size(),
            is_train=False,
        ),
    })
    
    if args.enable_flops_profiler:
        logging_rank_0(f"Enable DeepspeedFlopsProfiler!")
        deepspeed_flop_timers = DeepspeedFlopsTimerGroup({
            GENERATE:       FlopsProfiler(model=actor),
            ACTOR_TRAIN:    FlopsProfiler(model=actor),
            ACTOR_INFER:    FlopsProfiler(model=actor),
            # REF:            FlopsProfiler(model=im),
            CRITIC_TRAIN:   FlopsProfiler(model=critic),
            CRITIC_INFER:   FlopsProfiler(model=critic),
            # RM:             FlopsProfiler(model=reward_model),
        })
    else:
        logging_rank_0(f"Disable DeepspeedFlopsProfiler!")
        deepspeed_flop_timers = None

    # 初始化experience_maker replay_buffer
    
    if args.use_guide_action:
        logging_rank_0("Use Guide Action.", "debug")
    else:
        logging_rank_0("NOT Use Guide Action.", "debug")

    generate_kwargs = {
        "do_sample": True,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "bad_words_ids": bad_words_ids,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "use_cache": True,
        "pad_token_id": policy_tokenizer.pad_token_id,
        "eos_token_id": policy_tokenizer.eos_token_id,
    }
    
    experience_maker_args = {
        "actor": actor, "critic": critic, "reward_model": reward_model, "initial_model": im,
        "gen_args": generate_kwargs,
        "rm_granularity": rm_granularity,
        "ppo_granularity": ppo_granularity,
        "seed": args.seed + mpu.get_data_parallel_rank(),
        "kl_coef": args.kl_coef,
        "pad_token_id": policy_tokenizer.pad_token_id,
        "max_seq_len": args.policy_max_seq_len,
        "actor_minibatch_size": args.policy_minibatch_size,
        "rm_minibatch_size": args.rm_minibatch_size,
        "gen_minibatch_size": args.generate_minibatch_size,
        "gs_args": None,
        "enable_reward_scaling": args.enable_reward_scaling,
        "reward_scaling_gamma": args.gamma,
        "use_delta_reward": False,
        "enable_rm_lora": args.enable_rm_lora,
        "enable_policy_lora": args.enable_policy_lora,
        "logger": logger,
        "flops_timers": flop_timers,
        "deepspeed_flops_timers": deepspeed_flop_timers
    }
    
    # Step-level PPO
    if ppo_granularity is ActorGranularity.step:
        experience_maker_args["gs_args"] = GSArgs(
            enabling_tot=args.enabling_tot,
            enabling_bon=args.enabling_bon,
            gs_eval_batch_size=args.gs_eval_batch_size,
            gs_gen_batch_size=args.gs_gen_batch_size,
            gs_gen_repeat_times=args.gs_gen_repeat_times,
            gs_breadth=args.gs_breadth,
            gs_iterations=args.gs_iterations,
            best_of_n_times=args.best_of_n_times,
            min_step_lengths=args.min_step_lengths,
            generator_tokenizer=policy_tokenizer,
            evaluator_tokenizer=rm_tokenizer,
        )
        experience_maker = StepLevelExperienceMaker(**experience_maker_args)
        
        
    # Token-level PPO & Sample-level PPO
    else:
        experience_maker_args["use_guide_action"] = args.use_guide_action
        experience_maker = PPOPPExperienceMaker(**experience_maker_args)

    replay_buffer = NaiveReplayBuffer(
        sample_batch_size=args.sample_batch_size,
        limit=args.buffer_limit_size,
        cpu_offload=args.replay_buffer_cpu_offload,
    )
    
    #### 加载数据 ####
    free_task_to_prompts, guide_task_to_prompts, val_prompts, prompt_to_guidance = load_dateset(args)
    free_task_info, guide_task_info = "Free Tasks: ", "Guide Tasks: "
    for task_name, queries in free_task_to_prompts.items():
        free_task_info += f"{task_name}--{len(queries)}, "
    for task_name, queries in guide_task_to_prompts.items():
        guide_task_info += f"{task_name}--{len(queries)}, "
    
    logging_rank_0("#"*50)
    logging_rank_0(free_task_info)
    logging_rank_0(guide_task_info)
    logging_rank_0(f"Validate every {args.val_every_n_episode} episode.")
    logging_rank_0("#"*50)

    # 初始化trainer
    trainer_args = {
        "actor": actor,
        "critic": critic,
        "actor_optim": actor_optimizer,
        "critic_optim": critic_optimizer,
        "actor_lr_scheduler": actor_lr,
        "critic_lr_scheduler": critic_lr,
        "experience_maker": experience_maker,
        "replay_buffer": replay_buffer,
        "setup_dataloader_func": get_dataloader_build_func(
            args=args, tokenizer=policy_tokenizer, exp_save_path=exp_save_path, ppo_granularity=ppo_granularity
        ),
        "logger": logger,
        "ckpt_saving_func": get_save_checkpoint_callback(path=ckpt_save_path, save_optimizer=False),
        "rm_granularity": rm_granularity,
        "ppo_granularity": ppo_granularity,
        "eps_clip": args.eps_clip,
        "value_clip": args.value_clip,
        "drop_approx_kl": args.drop_approx_kl,
        "experience_batch_size": args.experience_batch_size,
        "max_epochs": args.max_epoch_per_update,
        "tokenizer": policy_tokenizer,
        "sample_replay_buffer": args.sample_replay_buffer,
        "entropy_loss_coef": args.entropy_loss_coef,
        "entropy_loss_decay_rate": args.entropy_loss_decay_rate,
        "clip_grad": args.clip_grad,
        "flops_timers": flop_timers,
        "deepspeed_flops_timers": deepspeed_flop_timers,
    }
    trainer_launch_args = {
        "prompts": free_task_to_prompts,
        "val_prompts": val_prompts,
        "seed": args.seed + mpu.get_data_parallel_rank(),
        "num_episodes": args.num_episodes,
        "max_timesteps": args.max_timesteps,
        "update_timesteps": args.update_timesteps,
        "val_check_interval": args.val_every_n_episode,
        "val_saving_func": get_val_experience_saving_func(args=args, tokenizer=policy_tokenizer, exp_save_path=exp_save_path, ppo_granularity=ppo_granularity)
    }
    
    if ppo_granularity is ActorGranularity.step:
        ppo_trainer = PPOTrainer(**trainer_args)
    else:
        if args.enable_constrain_actor:
            trainer_args.update({
                "constrain_actor": im,
                "constrain_actor_kl_coef": args.constrain_actor_kl_coef,
                "update_constrain_actor_interval": args.update_constrain_actor_interval,
                "target_constrain_actor_kl": args.target_constrain_actor_kl,
                "kl_adaptor_horizon": args.kl_adaptor_horizon,
            })
            if args.target_constrain_actor_kl is not None:
                logging_rank_0(f"Enable adaptive kl constrain coef. INIT={args.constrain_actor_kl_coef} TARGET={args.target_constrain_actor_kl} HORIZON={args.kl_adaptor_horizon}.")
            else:
                logging_rank_0(f"Enable fixed kl constrain coef. INIT={args.constrain_actor_kl_coef}.")
        ppo_trainer = PPOPPTrainer(**trainer_args)
        trainer_launch_args.update({
            "guided_prompts": guide_task_to_prompts,
            "guidance": prompt_to_guidance,
            "guidance_beta": args.ppopp_beta,
            "guidance_beta_decay": args.ppopp_beta_decay,
            "guidance_init_rate": args.ppopp_rate,
            "guidance_rate_decay": args.ppopp_rate_decay,
        })
        
    # 开始训练
    ppo_trainer.fit(**trainer_launch_args)
    return
