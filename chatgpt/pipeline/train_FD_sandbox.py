# encoding=utf-8
"""Launch Feedback Distillation Pipeline"""
import csv
import os
import socket
import jsonlines
from collections import defaultdict
from datetime import datetime
from typing import Tuple, Dict, Callable, List

import torch
from torch.utils.data import DataLoader, SequentialSampler

from fengshen_inner.models.megatron import mpu
from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM

from chatgpt.replay_buffer import FDReplayBuffer, DistributedBatchSampler
from chatgpt.experience_maker import (SandBoxExperienceMaker, SandBoxExperience, 
                                      SandBoxExperienceMakerV2, SandBoxExperienceV2)
from chatgpt.dataset import FeedbackDistillDataset
from chatgpt.nn.llama import LlamaActor, LlamaReflector
from chatgpt.strategies import (initialize_megatron, 
                                build_deepspeed_config,
                                setup_inference_engine,
                                setup_model_and_optimizer,
                                get_save_checkpoint_callback,
                                )
from chatgpt.trainer import FDTrainer, FDSandboxTrainer
from chatgpt.pipeline.config import ActorGranularity, FDPipelineConfig
from chatgpt.pipeline.tokenizer import TOKENIZER, FDPromptConvertion
from chatgpt.utils import logging_rank_0, local_rank, is_rank_0
from chatgpt.logger import WandbLogger
from chatgpt.pipeline.feedback_distill_template import actor_template


def load_dataset(args: FDPipelineConfig) -> List[Dict]:
    """
    Load FD_sandbox_data.

    """

    dataset_path = args.dataset_path

    with open(dataset_path, 'r') as file:
        file = list(jsonlines.Reader(file))

    return file


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

    def exp_saving_func(exp_dict, js, step):
        if is_rank_0():
            content = [k + ': \n' + str(v) + '\n\n' for k, v in exp_dict.items()]
            with open(os.path.join(exp_save_path, f'exp_step{step}.txt'), 'w') as file:
                file.writelines(content)
            if js is None:
                return
            with open(os.path.join(exp_save_path, f'exp_step{step}.json'), 'w') as file:
                file.write(js)

    return exp_saving_func


def launch_FD_sandbox(args: FDPipelineConfig) -> None:
    
    initialize_megatron(args=args)
    strategy = build_deepspeed_config(args=args)

    ### 准备用于保存的目录
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    running_name = f"FDSB-{args.wandb_group}-{args.wandb_name}-{socket.gethostname()}-{curr_time}"
    workspace_path = args.workspace_path
    exp_save_path = os.path.join(workspace_path, f"exp", running_name)
    os.makedirs(exp_save_path, exist_ok=True)

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
    logging_rank_0(f"actor pad_token_id: {actor_tokenizer.pad_token_id}.")

    ### 加载actor
    actor_model_path = args.policy_model_path
    actor_precision = torch.bfloat16 if args.policy_precision == 'bf16' else torch.float16
    actor_granularity = ActorGranularity[args.actor_granularity] # should be ActorGranularity.sample

    actor = LlamaActor(
        LlamaForCausalLM.from_pretrained(
            os.path.join(actor_model_path, f"part_{mpu.get_model_parallel_rank()}"), torch_dtype=actor_precision), 
        actor_granularity=actor_granularity, rm_granularity=None)
    actor, actor_optimizer, actor_lr = setup_model_and_optimizer(args, actor, strategy, learning_rate=args.actor_lr, scheduler_type=args.actor_scheduler_type)

    ### 初始化experience_maker和replay_buffer
    # experience_maker = SandBoxExperienceMaker(
    #     actor=actor,
    #     tokenizer=actor_tokenizer,
    #     template=actor_template
    # )

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

    experience_maker = SandBoxExperienceMakerV2(
        actor=actor,
        tokenizer=actor_tokenizer,
        template=actor_template,
        actor_gen_args=actor_gen_args,
        seed=args.seed + mpu.get_data_parallel_rank()
    )

    replay_buffer = FDReplayBuffer(
        sample_batch_size=args.sample_batch_size,
        limit=args.buffer_limit_size,
        cpu_offload=args.replay_buffer_cpu_offload,
        pad_token_id=actor_tokenizer.pad_token_id,
        exp_class=SandBoxExperienceV2
    )

    ### 加载数据
    dataset = load_dataset(args)

    ### 初始化trainer
    trainer_args = {
        "actor": actor,
        "actor_optim": actor_optimizer,
        "actor_lr_scheduler": actor_lr,
        "experience_maker": experience_maker,
        "replay_buffer": replay_buffer,
        "actor_tokenizer": actor_tokenizer,
        "setup_dataloader_func": get_dataloader_build_func(args=args),
        "divergence": args.divergence_type,
        "JSD_coef": args.JSD_coef,
        "GKD_coef": args.GKD_coef,
        "temperature": args.KD_temperature,
        "shrink": args.shrink,
        "level": args.level,
        "experience_batch_size": args.experience_batch_size,
        "max_epochs": args.max_epochs,
        "sample_replay_buffer": args.sample_replay_buffer,
        "exp_saving_func": get_experience_saving_func(exp_save_path),
        "logger": logger,
        "callbacks": [],
    }

    trainer_launch_args = {
        "prompts": dataset,
        "seed": args.seed + mpu.get_data_parallel_rank(),
        "num_episodes": args.num_episodes,
        "max_timesteps": args.max_timesteps,
        "update_timesteps": args.update_timesteps,
        "val_check_interval": args.val_every_n_episode,
    }
    FD_trainer = FDSandboxTrainer(**trainer_args)

    logging_rank_0(f"Start training.\n")
    FD_trainer.fit(**trainer_launch_args)
