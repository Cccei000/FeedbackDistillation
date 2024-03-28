import random
from typing import List, Tuple

import torch
import torch.distributed as dist
from chatgpt.experience_maker import Experience

from .base import ReplayBuffer
from .utils import BufferItem, make_experience_batch, split_experience_batch
from chatgpt.nn.utils import get_global_statistics
from chatgpt.utils import is_rank_0,print_rank_0


class NaiveReplayBuffer(ReplayBuffer):
    """Naive replay buffer class. It stores experience.

     Args:
         sample_batch_size (int): Batch size when sampling.
         limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
         cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True, pad_token_id:int=0) -> None:
        super().__init__(sample_batch_size, limit)
        self.cpu_offload = cpu_offload
        self.target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        
        self.items: List[BufferItem] = []
        # self.item_advantages: List[float] = []
        
        self.pad_token_id = pad_token_id
        return
    
    @torch.no_grad()
    def update_with_gae(self, gamma:float=1.0, lam:float=0.95, is_distributed:bool=True, eps:float=1e-8) -> Tuple[float, float]:
        
        all_gae_advantages = []
        
        for item in self.items:
            rewards = item.reward
            values = item.values
            act_msk = item.action_mask
            
            # print(f"R:{rewards}\nV:{values}")
            assert rewards.shape[0] > 1
            assert values.shape[0] > 1
            assert act_msk.shape[0] == rewards.shape[0] and act_msk.shape[0] == values.shape[0]
    
            lastgaelam = 0
            gae_advanteges = torch.zeros_like(values)
            gae_returns = torch.zeros_like(values)
            
            start_idx = (act_msk == 1).nonzero(as_tuple=True)[0][0].item()
            end_idx = (act_msk == 1).nonzero(as_tuple=True)[0][-1].item()
            
            for t in reversed(range(start_idx, end_idx + 1)):
                
                nextvalues = values[t + 1] if t < end_idx else 0.0
                delta = rewards[t] + gamma * nextvalues - values[t]
                lastgaelam = delta + gamma * lam * lastgaelam
                
                gae_advanteges[t] = lastgaelam
                gae_returns[t] = lastgaelam + values[t]
                all_gae_advantages.append(lastgaelam)

            item.advantages = gae_advanteges
            item.reward = gae_returns
            # print_rank_0("reward 2",item.reward)
            
        all_gae_advantages = torch.tensor(all_gae_advantages, device=self.target_device)
        
        if is_distributed and dist.is_initialized():
            global_mean, global_var, _ = get_global_statistics(all_gae_advantages)
        else:
            global_var, global_mean = torch.var_mean(all_gae_advantages)
        
        inverse_global_std = torch.rsqrt(global_var + eps)
        
        return global_mean.item(), inverse_global_std.item()

    @torch.no_grad()
    def update_with_step_gae(self, gamma:float=1.0, lam:float=0.95, is_distributed:bool=True, eps:float=1e-8) -> Tuple[float, float]:
        
        all_gae_advantages = []

        for item in self.items:
            rewards = item.reward
            values = item.values 
            act_msk = item.action_mask
            step_reward = item.step_reward 
            step_value = item.step_value 
            step_id = item.step_id
            
            assert step_reward.shape[0] == step_value.shape[0]
            lastgaelam = 0
            start_idx = step_reward.shape[0]-step_id.int()
            end_idx = step_reward.shape[0]

            for t in reversed(range(start_idx, end_idx)):
                nextvalues = step_value[t + 1] if t < end_idx-1 else 0.0
                delta = step_reward[t] + gamma * nextvalues - step_value[t]
                lastgaelam = delta + gamma * lam * lastgaelam
                last_return = lastgaelam + step_value[t]

            all_gae_advantages.append(lastgaelam)
            item.advantages = lastgaelam.unsqueeze(-1)
            item.reward = last_return

        all_gae_advantages = torch.tensor(all_gae_advantages, device=self.target_device)
        
        if is_distributed and dist.is_initialized():
            global_mean, global_var, _ = get_global_statistics(all_gae_advantages)
        else:
            global_var, global_mean = torch.var_mean(all_gae_advantages)
        
        inverse_global_std = torch.rsqrt(global_var + eps)
        
        return global_mean.item(), inverse_global_std.item()

    @torch.no_grad()
    def get_advantage_statistics(self,  is_distributed:bool=True, eps:float=1e-8) -> Tuple[float, float]:
        advantages = torch.tensor([item.advantages.item() for item in self.items], device=self.target_device)
        if is_distributed and dist.is_initialized():
            global_mean, global_var, _ = get_global_statistics(advantages)
        else:
            global_var, global_mean = torch.var_mean(advantages)
        
        inverse_global_std = torch.rsqrt(global_var + eps)

        return global_mean.item(), inverse_global_std.item()

    @torch.no_grad()
    def get_reward_statistics(self,  is_distributed:bool=True, eps:float=1e-8) -> Tuple[float, float]:
        rewards = torch.tensor([item.reward.item() for item in self.items], device=self.target_device)
        if is_distributed and dist.is_initialized():
            global_mean, global_var, _ = get_global_statistics(rewards)
        else:
            global_var, global_mean = torch.var_mean(rewards)
        
        global_std = torch.rsqrt(global_var + eps)

        return global_mean.item(), global_std.item()


    @torch.no_grad()
    def append(self, experiences: List[Experience]) -> None:
        for experience in experiences:
            if self.cpu_offload:
                experience.to_device(torch.device('cpu'))
            items = split_experience_batch(experience)
            self.items.extend(items)
            # self.item_advantages.extend([item.advantages.item() for item in items])
            if self.limit > 0:
                samples_to_remove = len(self.items) - self.limit
                if samples_to_remove > 0:
                    self.items = self.items[samples_to_remove:]
                    # self.item_advantages = self.item_advantages[samples_to_remove:]
        return

    def clear(self) -> None:
        self.items.clear()
        # self.item_advantages.clear()

    # FIXME: not support mp!!!
    @torch.no_grad()
    def sample(self) -> Experience:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items, sequence_padding_value=self.pad_token_id)
        if self.cpu_offload:
            experience.to_device(self.target_device)
        return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch, sequence_padding_value=self.pad_token_id)
        return experience
