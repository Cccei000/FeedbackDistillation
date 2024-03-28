import random
from typing import List, Dict, Tuple

import torch
import torch.distributed as dist
from chatgpt.experience_maker import PreferenceExperience

from .base import ReplayBuffer
from chatgpt.nn.utils import pad_3d_tensors


class PreferenceReplayBuffer(ReplayBuffer):
    """Preference replay buffer class. It stores preference experience.

     Args:
         sample_batch_size (int): Batch size when sampling.
         limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
         cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True, pad_token_id:int=0) -> None:
        super().__init__(sample_batch_size, limit)
        self.cpu_offload = cpu_offload
        self.target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        
        self.items: List[PreferenceExperience] = []
        # self.item_advantages: List[float] = []
        
        self.pad_token_id = pad_token_id
        return

    @torch.no_grad()
    def append(self, experiences: List[PreferenceExperience]) -> None:
        for experience in experiences:
            if self.cpu_offload:
                experience.to_device(torch.device('cpu'))
            self.items.append(experience)
            if self.limit > 0:
                samples_to_remove = len(self.items) - self.limit
                if samples_to_remove > 0:
                    self.items = self.items[samples_to_remove:]
        return

    def clear(self) -> None:
        self.items.clear()

    def _make_batch(self, samples):
        kwargs = {}
        to_pad_keys = set({'preference_sequences', 'ref_action_log_probs', 'action_mask', 'attention_mask'})
        # to_pad_keys = set({'preference_sequences', 'action_log_probs', 'ref_action_log_probs', 'action_mask', 'attention_mask'})
        for key in to_pad_keys:
            vals = [getattr(item, key) for item in samples]
            if key == "preference_sequences":
                batch_data, mask_3d, mask_2d = pad_3d_tensors(vals, padding_value=self.pad_token_id)
                kwargs['preference_mask'] = mask_2d
            else:
                batch_data, mask_3d, mask_2d = pad_3d_tensors(vals, padding_value=0.0)
 
            kwargs[key] = batch_data
        kwargs['task'] = [getattr(item, 'task') for item in samples]
        return kwargs 

    @torch.no_grad()
    def sample(self) -> Dict[str, torch.Tensor]:
        items = random.sample(self.items, self.sample_batch_size)
        data_dict = self._make_batch(items)
        if self.cpu_offload:
            for key,value in data_dict.items():
                data_dict[key] = value.to(self.target_device)
        return data_dict

    
    def get_advantage_statistics(self,  is_distributed:bool=True, eps:float=1e-8) -> Tuple[float, float]:
        pass
    
    def update_with_gae(self, gamma:float=1.0, lam:float=0.95, is_distributed:bool=True, eps:float=1e-8) -> Tuple[float, float]:
        pass

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> PreferenceExperience:
        return self.items[idx]

    def collate_fn(self, samples) -> Dict[str, torch.Tensor]:
        data_dict = self._make_batch(samples)
        return data_dict
