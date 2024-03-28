# encoding=utf-8
from typing import List, Tuple, Dict

import torch
import torch.distributed as dist
from chatgpt.experience_maker import Experience

from .base import ReplayBuffer
from .utils import BufferItem, make_experience_batch, split_experience_batch
from chatgpt.nn.utils import get_global_statistics
from chatgpt.replay_buffer import NaiveReplayBuffer


class LongTermReplayBuffer(NaiveReplayBuffer):
    
    def __init__(self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True, pad_token_id: int = 0) -> None:
        super().__init__(sample_batch_size, limit, cpu_offload, pad_token_id)
        
        self.prev_items: Dict[int, List[BufferItem]] = {}
        
        return
        
    @torch.no_grad()
    def append(self, experiences: List[Experience]) -> None:
        return super().append(experiences)
    
    def clear(self) -> None:
        return super().clear()
    
    @torch.no_grad()
    def sample(self) -> Experience:
        return super().sample()
    