# encoding=utf-8
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
from chatgpt.nn import Actor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Right padding for sequences is applied.

    Shapes of each tensor:
    sequences: (bs, seq_len).
    action_log_probs: (bs, seq_len-1)
    values: (bs,) or (bs, seq_len-1)
    reward: (bs,) or (bs, seq_len-1)
    advatanges: (bs) or (bs, seq_len-1)
    attention_mask: (bs, seq_len)
    action_mask: (bs, seq_len-1)
    origin_reward: (bs,) or (bs, seq_len-1)

    "bs": batch size.
    "seq_len": sequence length
    "seq_len-1": sequence length - 1, corresponding to seq[1:]
    """
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    reward: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor] = None
    action_mask: Optional[torch.BoolTensor] = None
    origin_reward: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.values = self.values.to(device)
        self.reward = self.reward.to(device)
        self.advantages = self.advantages.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)
        if self.origin_reward is not None:
            self.origin_reward = self.origin_reward.to(device)

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.action_log_probs = self.action_log_probs.pin_memory()
        self.values = self.values.pin_memory()
        self.reward = self.reward.pin_memory()
        self.advantages = self.advantages.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        if self.origin_reward is not None:
            self.origin_reward = self.origin_reward.pin_memory()
        return self


class ExperienceMaker(ABC):

    def __init__(self,
                 actor: Actor,
                 critic: nn.Module,
                 reward_model: nn.Module,
                 initial_model: Actor,
                 kl_coef: float = 0.1) -> None:
        """经验池采样抽象类

        Args:
            actor (Actor): _description_
            critic (nn.Module): _description_
            reward_model (nn.Module): _description_
            initial_model (Actor): _description_
            kl_coef (float, optional): _description_. Defaults to 0.1.
        """
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.initial_model = initial_model
        self.kl_coef = kl_coef

    @abstractmethod
    def make_experience(self, input_ids: torch.Tensor, **generate_kwargs) -> List[Experience]:
        pass

    @abstractmethod
    def logging(self):
        pass

@dataclass
class PreferenceExperience:
    """Experience is a batch of preference data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B) or (B, A)
    reward: (B) or (B, A)
    advatanges: (B)
    attention_mask: (B, S)
    action_mask: (B, A)

    "A" is the number of actions.
    """
    task: str
    preference_sequences: torch.Tensor
    ref_action_log_probs: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.preference_sequences = self.preference_sequences.to(device)
        # self.action_log_probs = self.action_log_probs.to(device)
        self.ref_action_log_probs = self.ref_action_log_probs.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.preference_sequences = self.preference_sequences.pin_memory()
        # self.action_log_probs = self.action_log_probs.pin_memory()
        self.ref_action_log_probs = self.ref_action_log_probs.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self

class PreferenceExperienceMaker(ABC):

    def __init__(self,
                 actor: Actor,
                 initial_model: Actor,
                 reward_model: nn.Module = None) -> None:
        super().__init__()
        self.actor = actor
        self.initial_model = initial_model
        self.reward_model = reward_model

    @abstractmethod
    def make_experience(self, input_ids: torch.Tensor, **generate_kwargs) -> List[PreferenceExperience]:
        pass

@dataclass
class StepExperience(Experience):
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.
    
    Shapes of each tensor (diff from Experience):
    step_id: (N,)
    step_reward: (N) or (N, STEPS)
    step_value: (N) or (N, STEPS)
    
    "N" equals to "bs"
    "STEPS": batch中单个样本拆分出的最大步数
    """
    step_reward: Optional[torch.Tensor] = None
    step_value: Optional[torch.Tensor] = None
    step_id: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        super().to_device(device)
        if self.step_reward is not None:
            self.step_reward = self.step_reward.to(device)
        if self.step_value is not None:
            self.step_value = self.step_value.to(device)
        if self.step_id is not None:
            self.step_id = self.step_id.to(device)
    def pin_memory(self):
        super().pin_memory()
        if self.step_reward is not None:
            self.step_reward = self.step_reward.pin_memory()
        if self.step_value is not None:
            self.step_value = self.step_value.pin_memory()
        if self.step_id is not None:
            self.step_id = self.step_id.pin_memory()
        return self
