from chatgpt.nn import Critic
from typing import Optional
import torch
from chatgpt.nn.utils import masked_mean

class GPTNeoXCritic(Critic):
    """
    GPT-NeoX Actor model.

    Args:
        model: Pretrained model
    """

    def __init__(self,
                 model) -> None:
        super().__init__(model)

    def forward(self,
                sequences: torch.LongTensor,
                action_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(sequences, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        values = self.value_head(last_hidden_states).squeeze(-1)[:, :-1] # 这里要错位，取state的隐层   # (bs, seq_len - 1)

        if action_mask is not None:
            num_actions = action_mask.size(1)
            values = values[:, -num_actions:]   # (bs, num_act)
            values = masked_mean(values, action_mask, dim=1) # (bs,)
            return values
        values = masked_mean(values, attention_mask[:,:-1], dim=1)  # (bs,)
        return values
