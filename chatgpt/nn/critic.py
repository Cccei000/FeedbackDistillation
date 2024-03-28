import math
from typing import Optional

import torch
import torch.nn as nn
from fengshen_inner.models.megatron import mpu
from fengshen_inner.models.megatron.layers.init_functions import orthogonal_init_method

# from .lora import LoRAModule
from .utils import masked_mean


class Critic(nn.Module):
    """
    Critic model base class.

    Args:
        model (nn.Module): Critic model.
        value_head (nn.Module): Value head to get value.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 model: nn.Module,
                 value_head: Optional[nn.Module] = None,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:
        """_summary_

        Args:
            model (nn.Module): _description_
            value_head (Optional[nn.Module], optional): _description_. Defaults to None.
            lora_rank (int, optional): _description_. Defaults to 0.
            lora_train_bias (str, optional): _description_. Defaults to 'none'.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        #super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        if value_head is not None:
            if value_head.out_features != 1:
                raise ValueError("The value head of critic model's output dim should be 1!")
            self.value_head = value_head
        else:
            self.value_head = mpu.RowParallelLinear(
                config=model.config,
                input_size=model.config.hidden_size,
                output_size=1,
                bias=False,
                input_is_parallel=False,
                init_method=orthogonal_init_method(),
                parallel_output=False,
                skip_bias_add=False,
            )
            # torch.manual_seed(42)
            # self.value_head = nn.Linear(model.config.hidden_size, 1)
            # nn.init.zeros_(self.value_head.bias)
            # nn.init.orthogonal_(self.value_head.weight, gain=math.sqrt(2))
        # self.convert_to_lora()

    def forward(self,
                sequences: torch.LongTensor,
                action_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(sequences, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']

        values = self.value_head(last_hidden_states)[0].squeeze(-1)[:, :-1]

        if action_mask is not None:
            num_actions = action_mask.size(1)
            values = values[:, -num_actions:]
            value = masked_mean(values, action_mask, dim=1)
            return value
        value = values.mean(dim=1).squeeze(1)
        return value
