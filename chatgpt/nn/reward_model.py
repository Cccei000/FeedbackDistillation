from typing import Optional

import torch
import torch.nn as nn

from .lora import LoRAModule,LoRAModule_Fenshen
from fengshen_inner.models.megatron import mpu



class RewardModel(LoRAModule_Fenshen):
    """
    Reward model base class.

    Args:
        model (nn.Module): Reward model.
        value_head (nn.Module): Value head to get reward score.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 model: nn.Module,
                 value_head: Optional[nn.Module] = None,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none', **kwargs) -> None:
        if lora_rank == 0:
            super().__init__()
        else:
            super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.body = model
        if value_head is not None:
            # if value_head.out_features != 1:
            #     raise ValueError("The value head of reward model's output dim should be 1!")
            self.value_head = value_head
        else:
            # self.value_head = nn.Linear(model.config.hidden_size, 1)
            self.value_head = mpu.RowParallelLinear(
                    config=model.config,
                    input_size=model.config.hidden_size,
                    output_size=1,
                    input_is_parallel=False,
                    skip_bias_add=False,
                    parallel_output=False,
                )            
        if lora_rank != 0:
            self.convert_to_lora()

    def forward(self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.body(sequences, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']
        values = self.value_head(last_hidden_states)[:, :-1]
        value = values.mean(dim=1).squeeze(1)    # ensure shape is (B)
        return value
