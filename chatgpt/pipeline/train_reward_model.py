# encoding=utf-8
import os
import warnings
from glob import glob

import datasets as ds
import torch
import torch.distributed.run
from fengshen_inner.models.llama.modeling_llama import LlamaModel
from fengshen_inner.models.megatron import mpu
from fengshen_inner.models.model_utils import (add_inverse_square_args,
                                               add_module_args)
from transformers import LlamaTokenizer

from chatgpt.dataset import RMCollator
from chatgpt.logger import WandbLogger
from chatgpt.nn.llama import LlamaRM, modeling_fengshenLlama_rm
from chatgpt.pipeline.config import RewardModelingPipelineConfig
from chatgpt.replay_buffer import DistributedBatchSampler
from chatgpt.strategies import (add_megatron_deepspeed_args,
                                build_deepspeed_config, initialize_megatron,
                                setup_model_and_optimizer)
from chatgpt.trainer import RMTrainer
from chatgpt.utils import is_rank_0, local_rank, logging_rank_0

from .utils import concat_prompt, load_jsonline_data, save_dataset_to_jsonl
from .tokenizer import TOKENIZER


def get_dataloader_build_func(args):
    def build_dataloader(dataset:ds.Dataset, collate_fn=None):
        rank = mpu.get_data_parallel_rank()
        # mp_rank = mpu.get_model_parallel_rank()
        # pp_rank = mpu.get_pipe_parallel_rank()
        world_size = mpu.get_data_parallel_world_size()
        global_batch_size = args.train_micro_batch_size_per_gpu * world_size

        # Use a simple sampler with distributed batch sampler
        dataset = dataset.shuffle(seed=42)
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
            collate_fn=collate_fn,
        )
    return build_dataloader

def get_save_checkpoint_callback(path:str):
    
    def saving(iteration, model, **kwargs):
        tag = f"global_step{iteration}"
        model.save_checkpoint(path, tag=tag)

    return saving


def get_del_checkpoint_callback(path:str):
    
    def deleting(iteration):
        try:
            if is_rank_0():
                tag = f"global_step{iteration}"
                ckpt_files = glob(os.path.join(path, f"{tag}/*"))
                for f in ckpt_files:
                    os.remove(f)
                os.rmdir(os.path.join(path, f"{tag}"))
        except:
            pass
        
        return
    return deleting


def calculate_total_steps(args:RewardModelingPipelineConfig, train_dataset_length:int):
    total_devices = args.gpus/int(args.tensor_model_parallel_size)
    train_batches = train_dataset_length// args.train_micro_batch_size_per_gpu // total_devices
    train_steps = (args.max_epochs * train_batches) // args.gradient_accumulation_steps
    logging_rank_0(f"Estimated stepping batches: {train_steps}")
    return train_steps

def launch_reward_modeling(args: RewardModelingPipelineConfig) -> int:
    """Main training program.

    Args:
        args (RewardModelingPipelineConfig): config

    Returns:
        int: best step
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    
    initialize_megatron(args)
    strategy = build_deepspeed_config(args)
    tokenizer = TOKENIZER[args.reward_model_type](tokenizer_path=args.rm_tokenizer_path)
    args.train_micro_batch_size_per_gpu=args.rm_batch_size
    
    ##### 准备用于保存的目录 #####
    # root
    workspace_path = args.workspace_path
    try:
        if not os.path.exists(workspace_path):
            os.makedirs(workspace_path)
    except:
        pass
    # splited data
    data_save_path = os.path.join(workspace_path, f"rm_data")
    try:
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
    except:
        pass
    # if not os.path.exists(os.path.join(data_save_path, "train")):
    #     os.mkdir(os.path.join(data_save_path, "train"))
    # if not os.path.exists(os.path.join(data_save_path, "eval")):
    #     os.mkdir(os.path.join(data_save_path, "eval"))
    # if not os.path.exists(os.path.join(data_save_path, "test")):
    #     os.mkdir(os.path.join(data_save_path, "test"))
    # ckpt
    ckpt_save_path = os.path.join(workspace_path, "ckpt/reward_model/")
    try:
        if not os.path.exists(ckpt_save_path):
            os.makedirs(ckpt_save_path)
    except:
        pass
    ##### Load dataset #####
    
    dataset = load_jsonline_data(args.dataset_path, prefix=args.prefix, seperator=args.multiturn_seperator)
    dataset = concat_prompt(dataset, prefix=args.prefix, seperator=args.multiturn_seperator)
    train_test_dataset = dataset.train_test_split(test_size=args.data_split_ratio["test"], shuffle=True, seed=args.seed)
    train_val_dataset = train_test_dataset["train"].train_test_split(test_size=args.data_split_ratio["eval"] / (1-args.data_split_ratio["test"]), shuffle=True, seed=args.seed)
    train_dataset = train_val_dataset["train"]
    eval_dataset = train_val_dataset["test"]
    test_dataset = train_test_dataset["test"]
    save_dataset_to_jsonl(train_dataset, os.path.join(data_save_path, "train.jsonl"))
    save_dataset_to_jsonl(eval_dataset, os.path.join(data_save_path, "eval.jsonl"))
    save_dataset_to_jsonl(test_dataset, os.path.join(data_save_path, "test.jsonl"))

    args.total_steps = calculate_total_steps(args, len(train_dataset)) if args.total_steps is None else args.total_steps

    # load reward model
    rm_precision = torch.bfloat16 if args.rm_precision == "bf16" else torch.float16 
    if args.from_sft:
        sft_model_path =  os.path.join(workspace_path, "models/policy")
        model = LlamaModel.from_pretrained(
            os.path.join(sft_model_path,f"part_{mpu.get_model_parallel_rank()}"),
            torch_dtype=torch.bfloat16
        )
        # model.is_bidirectional = True
        reward_model = LlamaRM(
            model=model,
            value_head=None,
            lora_rank=args.lora_rank,
            rm_granularity = args.rm_granularity,
            output_granularity = args.rm_granularity,
        ).to(dtype=rm_precision)
    else:
        rm_model_path = os.path.join(workspace_path, f"models/reward_model/")
        reward_model = modeling_fengshenLlama_rm(
            pretrained_path=os.path.join(rm_model_path, f"part_{mpu.get_model_parallel_rank()}"), lora_rank=args.lora_rank,
            rm_granularity = args.rm_granularity,
            actor_granularity = args.rm_granularity,
        ).eval().to(dtype=rm_precision).cpu()
    if args.activation_checkpointing:
        reward_model.gradient_checkpointing_enable()
        logging_rank_0(f"Set Gradient Checkpointing")
        
    reward_model, rm_optimizer, rm_lr = setup_model_and_optimizer(args, reward_model, strategy)
    
    # 初始化logger
    logger = WandbLogger(
        project=args.wandb_project,
        group=args.wandb_group,
        entity=args.wandb_team,
        name=f"{args.wandb_name}-reward-modeling",
        ignore=local_rank() != 0,
    )
    logger.log_hyperparams(args)

    collate_fn = {
        "train": RMCollator(tokenizer=tokenizer, max_length=args.rm_max_seq_len, query_key="query", response_key="preference"), #, prefix_bot="", prefix_user=""
        "eval": RMCollator(tokenizer=tokenizer, max_length=args.rm_max_seq_len, query_key="query", response_key="preference"), #, prefix_bot="", prefix_user=""
        "test": RMCollator(tokenizer=tokenizer, max_length=args.rm_max_seq_len, query_key="query", response_key="preference"), #, prefix_bot="", prefix_user=""
    }

    # 初始化trainer
    rm_trainer = RMTrainer(
        model=reward_model,
        optim=rm_optimizer,
        lr_scheduler=rm_lr,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        setup_dataloader_func=get_dataloader_build_func(args),
        max_epochs=args.max_epochs,
        logger=logger,
        collate_fn=collate_fn,
        val_check_interval=args.val_check_interval,
        ckpt_saving_func=get_save_checkpoint_callback(path=ckpt_save_path),
        save_best_n_ckpt=args.save_best_n_ckpt,
        ckpt_deleting_func=get_del_checkpoint_callback(path=ckpt_save_path),
        granularity=args.rm_granularity
    )
    # 开始训练
    best_step = rm_trainer.fit()

    return best_step

