# encoding=utf-8
import torch
from typing import Optional

import torch.distributed as dist
from fengshen_inner.models.megatron import mpu

from chatgpt.utils import logging_rank_0

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, device:torch.device):  # shape:the dimension of input data
        self.device = device
        
        ### DP rank 同步部分
        ### 如果没有DP，该部分维持默认值，不影响计算
        self.n_sync = torch.zeros(1, device=device, dtype=torch.int64)
        self.mean_sync = torch.zeros(shape, device=device, dtype=torch.float32)
        self.S_sync = torch.zeros(shape, device=device, dtype=torch.float32)
        self.std_sync = torch.sqrt(self.S_sync)
        
        ### DP rank 异步部分
        self.n_async = torch.zeros(1, device=device, dtype=torch.int64)
        self.mean_async = torch.zeros(shape, device=device, dtype=torch.float32)
        self.S_async = torch.zeros(shape, device=device, dtype=torch.float32)
        self.std_async = torch.sqrt(self.S_async)

    def update(self, x: torch.Tensor):
        x = x.clone().detach()
        
        ### 更新异步部分
        self.n_async += 1
        if self.n_async == 1:
            self.mean_async = x
            self.S_async = x ** 2
            self.std_async = x
        else:
            old_mean = self.mean_async.detach().clone()
            self.mean_async = old_mean + (x - old_mean) / self.n_async
            self.S_async = self.S_async + (x - old_mean) * (x - self.mean_async)
            self.std_async = torch.sqrt(self.S_async / self.n_async)
            
        # print(f"Rank-{self.device}: {self.n_async.item()} ({self.n_sync.item()}) ({self.n.item()}) | {self.mean_async.item()} ({self.mean_sync.item()}) ({self.mean.item()}) | {self.std_async.item()} ({self.std_sync.item()}) ({self.std.item()})")
    
    def update_batch(self, x_batch:torch.Tensor, value_mask:Optional[torch.Tensor]=None):
        
        if value_mask is None:
            batch_n = x_batch.numel()
            batch_mean = torch.mean(x_batch)
            batch_std = torch.std(x_batch) if batch_n > 1 else torch.tensor([1], dtype=x_batch.dtype, device=x_batch.device)
            # assert batch_n > 0
        else:
            batch_n = torch.sum(value_mask)
            # assert batch_n.item() > 0
            batch_mean = torch.sum(x_batch* value_mask) / batch_n
            if batch_n > 1:
                batch_var = torch.sum(((x_batch - batch_mean) ** 2) * value_mask) / batch_n
                batch_std = torch.sqrt(batch_var)
            else:
                batch_std = torch.tensor([1], dtype=x_batch.dtype, device=x_batch.device)
        batch_S = (batch_std ** 2) * batch_n
        merged_n = self.n_async + batch_n
        merged_mean = self.mean_async + batch_n * (batch_mean - self.mean_async) / (self.n_async + batch_n)
        merged_S = self.S_async + batch_S + (self.n_async + batch_n) * ((self.mean_async + batch_mean) ** 2) / (self.n_async + batch_n)
        # logging_rank_0(f"n: {merged_n}| mean: {merged_mean} | S: {merged_S}", "debug")
        self.n_async = merged_n
        self.mean_async = merged_mean
        self.S_async = merged_S
        
        return
    
    def sync(self):
        
        if dist.is_initialized():
            ### gather
            dp_world_size = mpu.get_data_parallel_world_size()
            synced_n = [torch.zeros_like(self.n_async, device=self.device) for _ in range(dp_world_size)]
            synced_mean = [torch.zeros_like(self.mean_async, device=self.device) for _ in range(dp_world_size)]
            synced_S = [torch.zeros_like(self.S_async, device=self.device) for _ in range(dp_world_size)]
            dist.all_gather(synced_n, self.n_async, group=mpu.get_data_parallel_group())
            dist.all_gather(synced_mean, self.mean_async, group=mpu.get_data_parallel_group())
            dist.all_gather(synced_S, self.S_async, group=mpu.get_data_parallel_group())
            
            ### merge
            for new_n, new_mean, new_S in zip(synced_n, synced_mean, synced_S):
                merged_n  = self.n_sync + new_n
                merged_mean =  self.mean_sync + new_n * (new_mean - self.mean_sync) / (self.n_sync + new_n)
                merged_S = self.S_sync + new_S + (self.n_sync * new_n) * ((self.mean_sync + new_mean) ** 2) / (self.n_sync + new_n)
                self.n_sync = merged_n
                self.mean_sync = merged_mean
                self.S_sync = merged_S
            self.std_sync = torch.sqrt(self.S_sync / self.n_sync)
            
            ### 重置 DP rank 异步部分
            self.n_async = torch.zeros_like(self.n_async, device=self.device)
            self.mean_async = torch.zeros_like(self.mean_async, device=self.device)
            self.S_async = torch.zeros_like(self.S_async, device=self.device)
            self.std_async = torch.sqrt(self.S_async)    
        
        return
    
    @property
    def n(self) -> torch.Tensor:
        # n = n_1 + n_2
        return self.n_sync + self.n_async
    
    @property
    def mean(self) -> torch.Tensor:
        # m = m_1 + ( n_2 / (n_1 + n_2)) * (m_2 - m_1)
        return self.mean_sync + self.n_async * (self.mean_async - self.mean_sync) / self.n
    
    @property
    def S(self) -> torch.Tensor:
        # n*var = S = S_1 + S_2 + (( n_1 * n_2) / ( n_1 + n_2 )) * (( m_1 + m_2 ) ** 2)
        return self.S_sync + self.S_async + (self.n_sync * self.n_async) * ((self.mean_sync + self.mean_async) ** 2) / self.n
    
    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.S / self.n)
    

class RewardScaling:
    
    def __init__(self, device:torch.device, shape:int=1, gamma:float=1.0):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape, device=device)
        self.R = torch.zeros(self.shape, device=device)

    def __call__(self, x):
        self.R = self.gamma * self.R + x.detach().clone()
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):
        self.R = torch.zeros_like(self.R)
        
    def sync(self):
        self.running_ms.sync()
    
    @property
    def mean(self):
        return self.running_ms.mean
    
    @property
    def std(self):
        return self.running_ms.std

    @property
    def n(self):
        return self.running_ms.n
