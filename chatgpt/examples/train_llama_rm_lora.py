# encoding=utf-8

"""Train"""
import os
import torch
import torch.distributed.run
import argparse
import datasets as ds
import wandb
import copy

from fengshen_inner.models.model_utils import add_module_args, add_inverse_square_args
from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel
from fengshen_inner.models.megatron import mpu
from fengshen_inner.models.llama.modeling_llama_lora import LlamaModelLora


from chatgpt.nn.llama import LlamaRM, modeling_llama_rm,LlamaFSRewardModel,LlamaRM_para
from chatgpt.trainer import RMTrainer
from chatgpt.logger import WandbLogger
from chatgpt.utils import local_rank, print_rank_0
from chatgpt.strategies import add_megatron_deepspeed_args, initialize_megatron, build_deepspeed_config, setup_model_and_optimizer
from chatgpt.dataset import RMCollator,RMCollator_MultiTurn
from chatgpt.replay_buffer import DistributedBatchSampler
from chatgpt.nn.llama import modeling_fengshenLlama_rm

from transformers import LlamaTokenizer


def add_neox_ppo_pipeline_args(parent_args:argparse.ArgumentParser):
    
    group = parent_args.add_argument_group("Experiment Args")
    group.add_argument("--wandb_project", type=str, default="RM_LLAMA")
    group.add_argument("--wandb_group", type=str, default="")
    group.add_argument("--wandb_team", type=str, default=None)

    group = parent_args.add_argument_group("Data Args")
    group.add_argument("--dataset_path", type=str, default= None)
    group.add_argument("--prefix_user", type=str, default="<human>:")
    group.add_argument("--prefix_bot", type=str, default="\n<bot>:")


    group = parent_args.add_argument_group("Trainer Args")
    group.add_argument("--num_workers", type=int, default=2)
    group.add_argument("--total_steps", type=int, default=None)
    group.add_argument("--rm_batch_size", type=int, default=1)
    group.add_argument("--val_check_interval", type=float, default=0.05)
    group.add_argument("--max_epochs", type=int, default=1)
    group.add_argument("--rm_ckpt_path", type=str, default=None)
    group.add_argument("--from_ckpt", type=str, default=None)
    # group.add_argument("--load_from_rm", action="store_true")
    group.add_argument("--activation_checkpointing", action="store_true")

    group = parent_args.add_argument_group("Model Args")
    group.add_argument("--rm_tokenizer_path", type=str, default=None) # tokenizer路径
    group.add_argument("--model_type", type=str, default="llama_13b") # tokenizer路径
    group.add_argument("--rm_model_path", type=str, default=None, help="rm ckpt路径") # 需要使用gpt-neox/utils中的gxy写的转换脚本转换参数的key
    group.add_argument("--max_length", type=int, default=1024)
    # group.add_argument("--lora_rank", type=int, default=0) # 0表示不使用lora

    group.add_argument("--lora_alpha", type=int, default=64)
    group.add_argument("--lora_r", type=int, default=16)
    group.add_argument("--lora_dropout", type=float, default=0.05)    
    return parent_args
    
def set_tokenizer(tokenizer_path, model_type):
    from tokenizers import AddedToken
    if model_type == "llama_13b":
        human = AddedToken("<human>",lstrip=False,rstrip=False,single_word=False,normalized=True)
        bot = AddedToken("<bot>",lstrip=False,rstrip=False,single_word=False,normalized=True)
        llama_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        llama_tokenizer.add_special_tokens({"additional_special_tokens":[human, bot]})
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
    else:
        llama_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        llama_tokenizer.pad_token_id = 3
    return llama_tokenizer

def get_dataloader_build_func(args):
    def build_dataloader(dataset:ds.Dataset, collate_fn=None):
        rank = mpu.get_data_parallel_rank()
        mp_rank = mpu.get_model_parallel_rank()
        pp_rank = mpu.get_pipe_parallel_rank()
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

def get_save_checkpoint_callback(args):
    
    def saving(iteration, model, **kwargs):
        tag = f"global_step{iteration}"
        model.save_checkpoint(args.rm_ckpt_path, tag=tag)

    return saving

def calculate_total_steps(args,train_dataset_length):
    total_devices = 8/int(args.tensor_model_parallel_size) #假设8卡
    train_batches = train_dataset_length// args.train_micro_batch_size_per_gpu // total_devices
    train_steps = (args.max_epochs * train_batches) // args.gradient_accumulation_steps
    print("estimated stepping batches:",train_steps)
    return train_steps

def launch(args):
    """
    Main training program.
    """
    train_dataset = ds.load_from_disk(args.dataset_path+f'/train')
    eval_dataset = ds.load_from_disk(args.dataset_path+f'/dev')
    test_dataset = ds.load_from_disk(args.dataset_path+f'/test')
    args.total_steps = calculate_total_steps(args,len(train_dataset)) if args.total_steps is None else args.total_steps

    initialize_megatron(args)
    strategy = build_deepspeed_config(args)
    tokenizer = set_tokenizer(args.rm_tokenizer_path,args.model_type)
    
    # load from sft FSmodel model_parallel
    base_model = LlamaModel.from_pretrained(
            f"{args.rm_model_path}/part_{mpu.get_model_parallel_rank()}",
            torch_dtype=torch.bfloat16
        ).to(dtype=torch.bfloat16)
    rm_config = copy.deepcopy(base_model.config)
    rm_config.torch_dtype = torch.bfloat16
    rm_config.lora = True
    rm_config.lora_alpha = args.lora_alpha
    rm_config.lora_r = args.lora_r
    rm_config.lora_dropout = args.lora_dropout
    print("rm_config",rm_config)
    reward_model =  LlamaModelLora(rm_config, base_model).to(dtype=torch.bfloat16)
    reward_model.enable_lora()
    reward_model = LlamaRM_para(
            model=reward_model,
            value_head=None,
        ).to(dtype=torch.bfloat16)
    reward_model, rm_optimizer, rm_lr = setup_model_and_optimizer(args, reward_model, strategy)

    print("local_rank: ",local_rank())
    # 初始化logger
    logger = WandbLogger(
        project=args.wandb_project,
        group=args.wandb_group,
        entity=args.wandb_team,
        ignore=local_rank() != 0,
    )
    logger.log_hyperparams(args)


    collate_fn = {
        "train":RMCollator_MultiTurn(tokenizer,args.max_length, prefix_user=args.prefix_user, prefix_bot=args.prefix_bot),
        "eval":RMCollator_MultiTurn(tokenizer,args.max_length, prefix_user=args.prefix_user, prefix_bot=args.prefix_bot),
        "test":RMCollator_MultiTurn(tokenizer,args.max_length, prefix_user=args.prefix_user, prefix_bot=args.prefix_bot),
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
        ckpt_saving_func=get_save_checkpoint_callback(args),
    )
    # 开始训练
    rm_trainer.fit()
    rm_trainer.test()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_module_args(parent_args=parser)
    parser = add_inverse_square_args(parent_args=parser)
    parser = add_neox_ppo_pipeline_args(parent_args=parser)
    parser = add_megatron_deepspeed_args(parent_args=parser)
    args = parser.parse_args()
    args.train_micro_batch_size_per_gpu=args.rm_batch_size
    print(args)
    launch(args)
    print("training finished!")
