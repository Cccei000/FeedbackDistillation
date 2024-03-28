# encoding=utf-8
"""Launch Feedback Distillation Pipeline"""
import csv
import os
import socket
from collections import defaultdict
from datetime import datetime
from typing import Tuple, Dict, Callable, List

import torch
from torch.utils.data import DataLoader, SequentialSampler

from transformers import PreTrainedTokenizer

from fengshen_inner.models.megatron import mpu
from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM


from chatgpt.replay_buffer import FDReplayBuffer, DistributedBatchSampler
from chatgpt.experience_maker import FDExperienceMaker
from chatgpt.dataset import FeedbackDistillDataset
from chatgpt.nn.llama import LlamaActor, LlamaReflector
from chatgpt.strategies import (initialize_megatron, 
                                build_deepspeed_config,
                                setup_inference_engine,
                                setup_model_and_optimizer,
                                get_save_checkpoint_callback,
                                )
from chatgpt.trainer import FDTrainer
from chatgpt.pipeline.config import ActorGranularity, FDPipelineConfig
from chatgpt.pipeline.tokenizer import TOKENIZER, FDPromptConvertion
from chatgpt.utils import logging_rank_0, local_rank, is_rank_0
from chatgpt.logger import WandbLogger



def load_dataset(args: FDPipelineConfig) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Load TruthfulQA dataset.

    """

    train_ratio = args.dataset_train_ratio
    seed = args.seed
    dataset_path = args.dataset_path

    dataset = defaultdict(list)
    trainset = defaultdict(list)
    valset = defaultdict(list)

    with open(dataset_path, 'r' ) as datafile:
        reader = csv.DictReader(datafile)
        for line in reader:
            dataset[line['Category']].append(line['Question'].strip())

    for task_name, queries in dataset.items():

        task_size = len(queries)
        train_task_size = int(task_size * train_ratio)
        if train_task_size >= task_size:
            trainset[task_name].extend(queries)
        else:
            g = torch.Generator()
            g.manual_seed(seed + mpu.get_data_parallel_rank())
            random_idx = torch.randperm(task_size, generator=g).tolist()
            queries = [queries[i] for i in random_idx]
            trainset[task_name].extend(queries[:train_task_size])
            valset[task_name].extend(queries[train_task_size:])

    return trainset, valset


def get_dataloader_build_func(args: FDPipelineConfig) -> Callable:

    def build_dataloader(replay_buffer: FDReplayBuffer) -> DataLoader:

        batch_size = args.actor_train_batch_size
        num_workers = args.num_workers

        rank = mpu.get_data_parallel_rank()
        mp_rank = mpu.get_model_parallel_rank()
        pp_rank = mpu.get_pipe_parallel_rank()
        world_size = mpu.get_data_parallel_world_size()
        global_batch_size = batch_size * world_size

        dataset = FeedbackDistillDataset(replay_buffer=replay_buffer)
        sampler = SequentialSampler(dataset)
        batch_sampler = DistributedBatchSampler(
            sampler=sampler,
            batch_size=global_batch_size,
            drop_last=True,
            rank=rank,
            world_size=world_size
        )

        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn
        )
        return dataloader
    
    return build_dataloader


def get_experience_saving_func(exp_save_path: str) -> Callable:

    def exp_saving_func(exp_dict, step):
        if is_rank_0():
            content = [k + ': \n' + str(v) + '\n\n' for k, v in exp_dict.items()]
            with open(os.path.join(exp_save_path, f'exp_step{step}.txt'), 'w') as file:
                file.writelines(content)

    return exp_saving_func


def get_val_experience_saving_func(args: FDPipelineConfig, tokenizer: PreTrainedTokenizer, exp_save_path: str) -> Callable:

    logging_rank_0("Validation experiences saving is not implemented yet.", level='info')

    save_val_experience = None

    return save_val_experience


def launch_FD(args: FDPipelineConfig) -> None:
    
    initialize_megatron(args=args)
    strategy = build_deepspeed_config(args=args)

    ### 准备用于保存的目录
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    running_name = f"FD-{args.wandb_name}-{socket.gethostname()}-{curr_time}"
    # root
    workspace_path = args.workspace_path
    logging_path = args.logging_path
    os.makedirs(workspace_path, exist_ok=True)
    # exp
    exp_save_path = os.path.join(workspace_path, f"exp", running_name)
    os.makedirs(exp_save_path, exist_ok=True)
    # ckpt
    ckpt_save_path = os.path.join(workspace_path, f"ckpt", running_name)
    os.makedirs(ckpt_save_path, exist_ok=True)
    # tensorboard logs
    # tb_log_path = os.path.join(logging_path, f"runs", running_name)
    # os.makedirs(tb_log_path, exist_ok=True)

    ### 初始化logger
    logger = WandbLogger(
        project=args.wandb_project,
        group=args.wandb_group,
        entity=args.wandb_team,
        ignore=local_rank() != 0,
        name=running_name,
        tensorboard_dir=None
    )
    logger.log_hyperparams(args.__dict__)
    args_info = ['Arguments infomation:']
    for k, v in args.__dict__.items():
        args_info.append(f"{k}: {v}")
    args_info = '\n'.join(sorted(args_info))
    logging_rank_0(args_info)

    ### 加载tokenizer
    actor_tokenizer = TOKENIZER[args.policy_model_type](tokenizer_path=args.policy_tokenizer_path)
    actor_tokenizer.padding_side = 'right'
    actor_tokenizer.pad_token_id = 0 if actor_tokenizer.pad_token_id is None else actor_tokenizer.pad_token_id
    reflector_tokenizer = TOKENIZER[args.reflector_model_type](tokenizer_path=args.reflector_tokenizer_path)
    reflector_tokenizer.padding_side = 'right'
    reflector_tokenizer.pad_token_id = 0 if reflector_tokenizer.pad_token_id is None else reflector_tokenizer.pad_token_id
    logging_rank_0(f"actor pad_token_id: {actor_tokenizer.pad_token_id}, reflector pad_token_id: {reflector_tokenizer.pad_token_id}.")

    ### 加载actor
    actor_model_path = args.policy_model_path
    actor_precision = torch.bfloat16 if args.policy_precision == 'bf16' else torch.float16
    actor_granularity = ActorGranularity[args.actor_granularity] # should be ActorGranularity.sample

    actor = LlamaActor(
        LlamaForCausalLM.from_pretrained(
            os.path.join(actor_model_path, f"part_{mpu.get_model_parallel_rank()}"), torch_dtype=actor_precision), 
        actor_granularity=actor_granularity, rm_granularity=None)
    actor, actor_optimizer, actor_lr = setup_model_and_optimizer(args, actor, strategy, learning_rate=args.actor_lr, scheduler_type=args.actor_scheduler_type)

    ### 加载reflector
    reflector_precision = torch.bfloat16 if args.reflector_precision == 'bf16' else torch.float16
    reflector_model_path = args.reflector_model_path
    reflector = LlamaReflector(
        LlamaForCausalLM.from_pretrained(
            os.path.join(reflector_model_path, f"part_{mpu.get_model_parallel_rank()}"), torch_dtype=reflector_precision))
    reflector = setup_inference_engine(model=reflector, mp_size=args.tensor_model_parallel_size, dtype=reflector_precision)

    ### 初始化experience_maker和replay_buffer
    actor_gen_args = {
        "do_sample": args.actor_do_sample,
        "top_p": args.actor_top_p,
        "top_k": args.actor_top_k,
        # "bad_words_ids": [],
        "max_new_tokens": args.actor_max_new_tokens,
        "repetition_penalty": args.actor_repetition_penalty,
        "temperature": args.actor_temperature,
        "use_cache": True,
        "pad_token_id": actor_tokenizer.pad_token_id,
        "eos_token_id": actor_tokenizer.eos_token_id,
    }

    reflector_gen_args = {
        "do_sample": args.reflector_do_sample,
        "top_p": args.reflector_top_p,
        "top_k": args.reflector_top_k,
        # "bad_words_ids": [],
        "max_new_tokens": args.reflector_max_new_tokens,
        "repetition_penalty": args.reflector_repetition_penalty,
        "temperature": args.reflector_temperature,
        "use_cache": True,
        "pad_token_id": reflector_tokenizer.pad_token_id,
        "eos_token_id": reflector_tokenizer.eos_token_id,
    }

    prompt_convertion = FDPromptConvertion(
        actor_tokenizer=actor_tokenizer,
        reflector_tokenizer=reflector_tokenizer,
        reflector_type=args.reflector_model_type)

    experience_maker = FDExperienceMaker(
        actor=actor,
        reflector=reflector,
        prompt_convertion=prompt_convertion,
        seed=args.seed + mpu.get_data_parallel_rank(),
        actor_minibatch_size=args.actor_mini_batch_size,
        reflector_minibatch_size=args.reflector_mini_batch_size,
        actor_gen_args=actor_gen_args,
        reflector_gen_args=reflector_gen_args,
    )

    replay_buffer = FDReplayBuffer(
        sample_batch_size=args.sample_batch_size,
        limit=args.buffer_limit_size,
        cpu_offload=args.replay_buffer_cpu_offload,
        pad_token_id=actor_tokenizer.pad_token_id
    )

    ### 加载数据
    training_queries, validation_queries = load_dataset(args)
    data_category_info = '\n' + 'Dataset category info'.ljust(40) + '[train|val]:\n'
    training_query_count, validation_query_count = 0, 0
    for task, queries in training_queries.items():
        train_size = len(queries)
        val_size = len(validation_queries.get(task, []))
        data_category_info += f'Task "{task}" size'.ljust(40) + f'[{train_size}|{val_size}]\n'
        training_query_count += train_size
        validation_query_count += val_size

    logging_rank_0("#"*50)
    logging_rank_0(data_category_info.rstrip())
    logging_rank_0(f"Total training queries: [{training_query_count}]")
    logging_rank_0(f"Total validation queries: [{validation_query_count}]")
    logging_rank_0("#"*50)

    ### 初始化trainer
    trainer_args = {
        "actor": actor,
        "actor_optim": actor_optimizer,
        "actor_lr_scheduler": actor_lr,
        "experience_maker": experience_maker,
        "replay_buffer": replay_buffer,
        "actor_tokenizer": actor_tokenizer,
        "setup_dataloader_func": get_dataloader_build_func(args=args),
        "constraint_actor": None,
        "divergence": args.divergence_type,
        "JSD_coef": args.JSD_coef,
        "GKD_coef": args.GKD_coef,
        "temperature": args.KD_temperature,
        "skip_exp": args.skip_exp,
        "shrink": args.shrink,
        "constrain_actor_kl_coef": None,
        "target_constrain_actor_kl": None,
        "kl_adaptor_horizon": None,
        "mixed_sampling": False,
        "separate_sampling": False,
        "clip_grad": False,
        "experience_batch_size": args.experience_batch_size,
        "max_epochs": args.max_epochs,
        "sample_replay_buffer": args.sample_replay_buffer,
        "update_constraint_actor_interval": 0,
        "ckpt_saving_func": get_save_checkpoint_callback(path=ckpt_save_path, save_optimizer=False),
        "exp_saving_func": get_experience_saving_func(exp_save_path),
        "logger": logger,
        "callbacks": [],
    } # to be continued

    trainer_launch_args = {
        "prompts": training_queries,
        "val_prompts": validation_queries,
        "seed": args.seed + mpu.get_data_parallel_rank(),
        "num_episodes": args.num_episodes,
        "max_timesteps": args.max_timesteps,
        "update_timesteps": args.update_timesteps,
        "val_check_interval": args.val_every_n_episode,
        "val_saving_func": get_val_experience_saving_func(
            args=args, tokenizer=actor_tokenizer, exp_save_path=exp_save_path),
    }
    FD_trainer = FDTrainer(**trainer_args)

    logging_rank_0(f"Start training.\n")
    FD_trainer.fit(**trainer_launch_args)
