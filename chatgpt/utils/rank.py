# encoding=utf-8
import os

import torch.distributed as dist
from fengshen_inner.models.megatron import mpu


def local_rank():
    """Local rank of process"""
    local_rank = os.environ.get("LOCAL_RANK")

    if local_rank is None:
        local_rank = os.environ.get("SLURM_LOCALID")

    if local_rank is None:
        print(
            "utils.local_rank() environment variable LOCAL_RANK not set, defaulting to 0",
            flush=True,
        )
        local_rank = 0
    return int(local_rank)

def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def is_all_rank_0() -> bool:
    if not dist.is_initialized():
        return True
    dp_rank = mpu.get_data_parallel_rank()
    mp_rank = mpu.get_model_parallel_rank()
    pp_rank = mpu.get_pipe_parallel_rank()
    
    return dp_rank == 0 and mp_rank == 0 and pp_rank == 0


def print_rank_0(*message):
    """If distributed is initialized print only on rank 0."""
    
    if is_rank_0():
        print(*message, flush=True)
        
    return
