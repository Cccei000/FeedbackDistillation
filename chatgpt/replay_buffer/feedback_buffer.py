from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
from collections import defaultdict
import random

import torch
import torch.nn.functional as F
from chatgpt.experience_maker import FDExperience, SandBoxExperience
from chatgpt.nn.utils import zero_pad_sequences
from .base import ReplayBuffer
from chatgpt.utils import logging_rank_0


@dataclass(kw_only=True)
class FDBufferItem:

    sequences: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    ref_sequences: torch.Tensor
    ref_attention_mask: torch.Tensor
    ref_action_mask: torch.Tensor
    scores: torch.Tensor
    repeating: torch.Tensor


class FDReplayBuffer(ReplayBuffer):

    """
     Args:
         sample_batch_size (int): Batch size when sampling.
         limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
         cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True, pad_token_id: int = 0, exp_class = FDExperience) -> None:
        super().__init__(sample_batch_size, limit)
        self.cpu_offload = cpu_offload       
        self.items: List[Dict] = []
        self.pad_token_id = pad_token_id
        self.exp_class = exp_class
        return

    @torch.no_grad()
    def append(self, experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        batch_size = experience.batchsize
        batch_kwargs = [{} for _ in range(batch_size)]
        for key, value in asdict(experience).items():
            assert batch_size == value.shape[0], f"experience.{key} has shape {value.shape}, but got batchsize {batch_size}"
            for i, v in enumerate(torch.unbind(value)):
                batch_kwargs[i][key] = v.unsqueeze(0)
        items = batch_kwargs
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    def _make_batch(self, items: List[Dict], pad_token_id: Optional[int] = None):
        pad_token_id = self.pad_token_id if pad_token_id is None else pad_token_id
        exp_kwargs = defaultdict(list)
        for item in items:
            for k, v in item.items():
                exp_kwargs[k].append(v)
        for k, v in exp_kwargs.items():
            if 'seq' in k:
                exp_kwargs[k] = zero_pad_sequences(v, side='right', padding_value=pad_token_id)
            else:
                exp_kwargs[k] = zero_pad_sequences(v, side='right', padding_value=False)
        return self.exp_class(**exp_kwargs)

    @torch.no_grad()
    def sample(self, sample_batch_size=None):
        if sample_batch_size is None:
            sample_batch_size = self.sample_batch_size
        items = random.sample(self.items, sample_batch_size)
        experience = self._make_batch(items, self.pad_token_id)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]

    def collate_fn(self, batch):
        return self._make_batch(batch, self.pad_token_id)
    
    def get_advantage_statistics(self,  is_distributed:bool=True, eps:float=1e-8) -> Tuple[float, float]:
        pass
    
    def update_with_gae(self, gamma:float=1.0, lam:float=0.95, is_distributed:bool=True, eps:float=1e-8) -> Tuple[float, float]:
        pass

