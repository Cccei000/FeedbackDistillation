from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from chatgpt.nn import Actor
from chatgpt.nn.utils import log_probs_from_logits


class GPTNeoXActor(Actor):
    """
    GPT-NeoX Actor model.

    Args:
        model: Pretrained model
    """

    def __init__(self,
                 model) -> None:
        super().__init__(model)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        return_action_mask: bool = True,
        **kwargs
    ) -> Union[Tuple[torch.LongTensor, torch.LongTensor], Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]:
        
        pad_token_id = kwargs.get('pad_token_id', None)
        input_attention_mask = input_ids.not_equal(pad_token_id).to(dtype=torch.bool, device=self.model.device)
        # print(f"OUT_MSK: {input_attention_mask} | IDS: {input_ids}")
        sequences = self.model.generate(
            input_ids.to(self.model.device), attention_mask=input_attention_mask, **kwargs
        )   # (bs, seq_len)

        attention_mask = None
        if pad_token_id is not None:
            attention_mask = sequences.not_equal(pad_token_id).to(dtype=torch.bool, device=sequences.device) # (bs, seq_len)
        if not return_action_mask:
            return sequences, attention_mask
        input_len = input_ids.size(1)
        eos_token_id = kwargs.get('eos_token_id', None)
        if eos_token_id is None:
            action_mask = torch.ones_like(sequences, dtype=torch.bool)  # (bs, seq_len)
        else:
            # left padding may be applied, only mask action
            action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0    # (bs, seq_len - input_len)
            action_mask = F.pad(action_mask, (1 + input_len, -1), value=True)               # (bs, seq_len + input_len) 这个(1 + input_len, -1)就很精髓！细品！！
        action_mask[:, :input_len] = False  # 将[0,input_len)的部分置零
        action_mask = action_mask[:, 1:]    # (bs, seq_len)
        return sequences, attention_mask, action_mask[:, -(sequences.size(1) - input_len):] # (bs, seq_len), (bs, seq_len), (bs, seq_len - input_len)

    def forward(self,
                sequences: torch.LongTensor,
                num_actions: int,
                attention_mask: Optional[torch.Tensor] = None,
                return_logits: bool = False) -> torch.Tensor:
        
        """Returns action log probs
        """
        # attention_mask = attention_mask[:, None, None, :]
        output = self.model(
            sequences, 
            attention_mask=attention_mask
        )
        logits = output[0]  # (bs, seq_len, vocab_size) 但是错位
        log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])  # (bs, seq_len - 1)

        if return_logits:
            return log_probs[:, -num_actions:], logits[:, -(num_actions + 1):-1, :]

        return log_probs[:, -num_actions:] # (bs, num_act)
