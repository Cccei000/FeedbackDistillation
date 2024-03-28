# encoding-utf-8
from typing import List

import torch
from torch.utils.data import Dataset

from chatgpt.experience_maker import Experience
from chatgpt.pipeline.config import ActorGranularity
from chatgpt.replay_buffer import (BufferItem, ReplayBuffer,
                                   make_experience_batch)
from chatgpt.utils import LoggingLevel, logging_rank_0


class ReplayBufferDataset(Dataset):


    def __init__(self,
                 replay_buffer:ReplayBuffer,
                 ppo_granularity:ActorGranularity,
                 is_distribted: bool = True,
                 norm_adv: bool = True,
                 pad_token_id:int=0,
                 gamma:float = 1.0,
                 lam:float = 0.95,
                 for_validation:bool=False,
                 **kwargs) -> None:
        """基于ReplayBuffer构建Dataset

        Args:
            replay_buffer (ReplayBuffer):       经验池
            ppo_granularity (ActorGranularity): PPO流程采用的粒度
            is_distribted (bool, optional):     是否启动分布式训练. Defaults to True.
            norm_adv (bool, optional):          是否对advantage进行normalization. Defaults to True.
            pad_token_id (int, optional):       Actor使用的pad token id. Defaults to 0.
            gamma (float, optional):            GAE gamma. Defaults to 1.0.
            lam (float, optional):              GAE lambda. Defaults to 0.95.
            for_validation（bool, optional):    当启用step level PPO且用于验证时，设置为 True。其原因为step level ppo在验证时不进行分步，而是评估完整的文本. Defaults to False.                              
        """        
        super().__init__()
        ### 参数检查 ###
        for key in kwargs:
            logging_rank_0(f"Deprecation: {key} has removed in 'ReplayBufferDataset'.", LoggingLevel.WARNING)
        
        self.replay_buffer = replay_buffer
        
        # 不进行 Advantage Normalization
        if not norm_adv:
            self.collate_fn = ReplayBufferCollator(ppo_granularity=ppo_granularity, adv_mean=0.0, inverse_adv_std=1.0, pad_token_id=pad_token_id)
            return
        
        # Sample-level PPO GAE
        if ppo_granularity is ActorGranularity.sample:
            advantage_mean, inverse_advantagea_std = replay_buffer.get_advantage_statistics(is_distributed=is_distribted)
        # Step_level PPO GAE
        elif ppo_granularity is ActorGranularity.step:
            # When doing validation
            if for_validation:
                advantage_mean, inverse_advantagea_std = replay_buffer.get_advantage_statistics(is_distributed=is_distribted)
            else:
                advantage_mean, inverse_advantagea_std = replay_buffer.update_with_step_gae(gamma=gamma, lam=lam, is_distributed=is_distribted)
        # Token-level PPO GAE
        else:
            advantage_mean, inverse_advantagea_std = replay_buffer.update_with_gae(gamma=gamma, lam=lam, is_distributed=is_distribted)
        
        self.collate_fn = ReplayBufferCollator(ppo_granularity=ppo_granularity, adv_mean=advantage_mean, inverse_adv_std=inverse_advantagea_std, pad_token_id=pad_token_id)

        return

    def __len__(self):
        return len(self.replay_buffer)
    
    def __getitem__(self, index):
        return self.replay_buffer[index]
    

class ReplayBufferCollator:

    def __init__(self,
                 ppo_granularity: ActorGranularity,
                 adv_mean:float = .0,
                 inverse_adv_std:float = 1.,
                 pad_token_id:int=0) -> None:
        """Collator for ReplayBufferDataset.

        Args:
            ppo_granularity (ActorGranularity): _description_
            adv_mean (float, optional):         ReplayBufferDataset中所有样本的advantage的均值. Defaults to .0.
            inverse_adv_std (float, optional):  ReplayBufferDataset中所有样本的advantage的方差的倒数. Defaults to 1..
            pad_token_id (int, optional):       Actor采用的pad token的id. Defaults to 0.
        """
        self.adv_mean = adv_mean
        self.inverse_adv_std = inverse_adv_std
        self.pad_token_id = pad_token_id
        self.ppo_granularity = ppo_granularity
        return

    def __call__(self, items: List[BufferItem]) -> Experience:
        """将经验样本的列表打包为 batch

        Args:
            items (List[BufferItem]): 经验样本的列表

        Returns:
            Experience: batch
        """
        experiences = make_experience_batch(items, sequence_padding_value=self.pad_token_id)
        
        # Sample-level PPO || Step-level PPO：所有值进行 normalization
        if self.ppo_granularity is ActorGranularity.sample or self.ppo_granularity is ActorGranularity.step:
            experiences.advantages = (experiences.advantages - self.adv_mean) * self.inverse_adv_std # （bs,)
        # Token-level PPO：action mask 下的值进行 normalization
        else:
            experiences.advantages = torch.where(
                experiences.action_mask,
                (experiences.advantages - self.adv_mean) * self.inverse_adv_std,
                experiences.advantages
            )   # (bs, padded_act_len)
        return experiences
