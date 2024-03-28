# encoding=utf-8
import os
import torch
import argparse
import deepspeed
from glob import glob

from fengshen_inner.models.megatron import fused_kernels, mpu
from fengshen_inner.strategies.megatron_deepspeed import DeepSpeedStrategy
from fengshen_inner.models.model_utils import inverse_square_root_schedule, get_scheduler

from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from transformers.optimization import AdamW
from chatgpt.utils import logging_rank_0


def add_megatron_deepspeed_args(parent_args:argparse.ArgumentParser):
    
    group = parent_args.add_argument_group("Megatron Args")
    group.add_argument("--tensor_model_parallel_size", type=int, default=1)
    group.add_argument("--pipe_model_parallel_size", type=int, default=1)
    group.add_argument("--seed", type=int, default=1234, help="seed")
    group.add_argument("--rank", type=int, default=0)
    group.add_argument("--world_size", type=int, default=64)
    
    group.add_argument("--policy_precision", type=str, default="fp16")
    group.add_argument("--deepspeed_stage", type=int, default=1)
    group.add_argument("--offload_optimizer", action="store_true", default=False)
    group.add_argument("--gradient_accumulation_steps", type=int, default=32)
    group.add_argument("--train_micro_batch_size_per_gpu", type=int, default=1)
    group.add_argument("--loss_scale", type=float, default=0)
    group.add_argument("--initial_scale_power", type=int, default=16)
    group.add_argument("--loss_scale_window", type=int, default=1000)
    group.add_argument("--hysteresis", type=int, default=2)
    group.add_argument("--min_loss_scale", type=int, default=2)
    
    return parent_args

def initialize_megatron(args):
    
    tensor_model_parallel_size = args.tensor_model_parallel_size
    pipe_model_parallel_size = args.pipe_model_parallel_size
    seed = args.seed

    fused_kernels.load_fused_kernels()

    deepspeed.init_distributed(
        dist_backend='nccl',
        distributed_port=os.getenv("MASTER_PORT", "6000"),
        verbose=False,
        # timeout=datetime.timedelta(seconds=3600),
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    device_count = torch.cuda.device_count()
    device = rank % device_count
    torch.cuda.set_device(device)
    
    logging_rank_0(f"Device count: {device_count} | Rank: {rank}")

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
    # mpu.set_model_parallel_rank(rank)


def get_default_update_params(pl_model: torch.nn.Module, weight_decay:float):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'layer_norm.', 'layernorm.']
    optimizer_grouped_params = [
        {'params': [p for n, p in pl_model.named_parameters() if not any(
            nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': weight_decay},
        {'params': [p for n, p in pl_model.named_parameters() if any(
            nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]
    return optimizer_grouped_params


def configure_optimizers(args, model, strategy=None, learning_rate=None, scheduler_type=None):
    '''
    Args:
    '''
    # get params that optimizer need
    optimizer_grouped_params = get_default_update_params(model, args.weight_decay)
    
    if learning_rate is None:
        learning_rate = args.learning_rate

    # Configure optimizer.
    if strategy is not None:
        if 'offload_optimizer' in strategy.config['zero_optimization']:
            logging_rank_0(f"Use optimizer: 'DeepSpeedCPUAdam'", "debug")
            optimizer = DeepSpeedCPUAdam(
                optimizer_grouped_params, adamw_mode=True,
                lr=learning_rate,
                betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon)
        else:
            logging_rank_0(f"Use optimizer: 'FusedAdam'", "debug")
            optimizer = FusedAdam(
                optimizer_grouped_params, adam_w_mode=True,
                lr=learning_rate,
                betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon)
    else:
        logging_rank_0(f"Use optimizer: 'AdamW'", "debug")
        optimizer = AdamW(optimizer_grouped_params, lr=learning_rate,
                          betas=(args.adam_beta1, args.adam_beta2),
                          eps=args.adam_epsilon)
    # Configure learning rate scheduler.
    total_steps = args.lr_decay_ratio * \
        args.total_steps if args.lr_decay_steps == 0 else args.lr_decay_steps
    warmup_steps = args.warmup_ratio * \
        args.total_steps if args.warmup_steps == 0 else args.warmup_steps

    scheduler_type = scheduler_type if scheduler_type is not None else args.scheduler_type
    
    if scheduler_type == "inverse_sqrt":
        scheduler = inverse_square_root_schedule(optimizer=optimizer,
                                                 num_warmup_steps=warmup_steps, lr_min=args.warmup_min_lr, lr_max=args.warmup_max_lr)
    else:
        scheduler = get_scheduler(name=scheduler_type, optimizer=optimizer,
                                  num_warmup_steps=warmup_steps, num_training_steps=total_steps,
                                  lr_end=args.min_learning_rate)
    # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    return optimizer, scheduler, optimizer_grouped_params


def get_total_params(model):
    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        params = sum([p.nelement() for p in model.parameters()])
        logging_rank_0(
            " > number of parameters on model parallel rank {}: {}".format(
                mpu.get_model_parallel_rank(), params
            ),
            "debug"
        )
    else:
        params = 0
    total_n_parameters = torch.tensor([params]).cuda(torch.cuda.current_device())

    torch.distributed.all_reduce(total_n_parameters)
    total_n_parameters = total_n_parameters.item()
    return total_n_parameters


def setup_model_and_optimizer(args, model, strategy, learning_rate=None, scheduler_type=None):
    '''
    返回model, optimizer, lr_scheduler
    '''

    optimizer, lr_scheduler, model_params = configure_optimizers(args, model, strategy, learning_rate, scheduler_type)
    
    logging_rank_0("DeepSpeed is enabled.", "debug")
    rank = torch.distributed.get_rank()
    num_gpus = torch.cuda.device_count()
    device_id = rank%num_gpus
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=argparse.Namespace(device_rank=device_id),
        model=model,
        optimizer=optimizer,
        config=strategy.config,
        lr_scheduler=lr_scheduler,
        dist_init_required=False,
        model_parameters=model_params,
        mpu=mpu,
    )
    model.total_params = get_total_params(model.module)
    logging_rank_0(f' > total params: {"{:,}".format(model.total_params)}', "debug")

    return model, optimizer, lr_scheduler

def setup_inference_engine(model, mp_size:int, dtype):
    
    model = deepspeed.init_inference(
        model,
        mp_size=mp_size,
        dtype=dtype,
        replace_with_kernel_inject=True,
    )
    
    return model

def get_save_checkpoint_callback(path, save_optimizer:bool=True):
    
    def saving(iteration, model:deepspeed.DeepSpeedEngine, **kwargs):
        tag = f"global_step{iteration}"
        model.save_checkpoint(path, tag=tag)
        
        # delete optimizer
        if not save_optimizer:
            try:
                opts = glob(os.path.join(path, tag, "bf*"))
                for opt in opts:
                    os.remove(opt)
            except:
                pass

    return saving


def build_deepspeed_config(args):
    
    # 使用DeepSpeedStrategy读取deepspeed配置
    strategy = DeepSpeedStrategy(
        zero_optimization=True,
        stage=args.deepspeed_stage,
        offload_optimizer=args.offload_optimizer,
        partition_activations=True,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipe_model_parallel_size=args.pipe_model_parallel_size,
        mpu_seed=args.seed,
    )
    strategy.config.setdefault("train_micro_batch_size_per_gpu", args.train_micro_batch_size_per_gpu)
    strategy.config["gradient_accumulation_steps"]  = args.gradient_accumulation_steps

    if args.policy_precision == "fp16":
        logging_rank_0("Enabling DeepSpeed FP16.", "debug")
        strategy.config["fp16"] = {
            "enabled": True,
            "loss_scale": args.loss_scale,
            "initial_scale_power": args.initial_scale_power,
            "loss_scale_window": args.loss_scale_window,
            "hysteresis": args.hysteresis,
            "min_loss_scale": args.min_loss_scale,
        }
    elif args.policy_precision == "bf16":
        logging_rank_0("Enabling DeepSpeed BF16.", "debug")
        strategy.config["bf16"] = {"enabled": True}
    
    if "activation_checkpointing" in strategy.config:
        strategy.config["activation_checkpointing"]["cpu_checkpointing"]=True
    
    try:
        if args.enable_hybrid_engine:
            deepspeed_ver = deepspeed.__version__.split(".")
            deepspeed_ver = [int(item) for item in deepspeed_ver]
            if deepspeed_ver[1] < 9 and deepspeed_ver[0] == 0:
                logging_rank_0("Can't Enable DeepSpeedHybridEngine. Requiring 'deepspeed>=0.9.0'.", "debug")
            else:
                logging_rank_0("Enabling DeepSpeedHybridEngine.", "debug")
                strategy.config["hybrid_engine"] = {
                    "enabled": True,
                    "max_out_tokens": args.max_new_tokens,
                    "inference_tp_size": args.tensor_model_parallel_size,
                    "release_inference_cache": False,
                    "pin_parameters": True
                }
    except:
        pass

    return strategy
