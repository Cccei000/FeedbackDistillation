from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generation import generate
# from .lora import LoRAModule
from .utils import log_probs_from_logits


class Actor(nn.Module):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self, model: nn.Module, lora_rank: int = 0, lora_train_bias: str = 'none') -> None:
        super().__init__()
        self.model = model
        # super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        # self.convert_to_lora()

    @torch.no_grad()
    def _get_first_not_pad_position(self, ids: torch.Tensor, pad_token_id: int): #[bs, seq_len]
        cumulative_sum = torch.cumsum(ids.not_equal(pad_token_id), dim=1)
        flipped_tensor = torch.flip(cumulative_sum == 0, dims=[1]).int()
        first_not_pad_pos = (flipped_tensor.shape[1] - flipped_tensor.argmax(dim=1)).min()
        return first_not_pad_pos

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        return_action_mask: bool = True,
        **kwargs
    ) -> Union[Tuple[torch.LongTensor, torch.LongTensor], Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]:

        sequences = generate(self.model, input_ids, **kwargs)
        input_len = input_ids.size(1)
        attention_mask = None
        pad_token_id = kwargs.get('pad_token_id', None)
        eos_token_id = kwargs.get('eos_token_id', None)
        if pad_token_id is not None:
            trunc_pos = self._get_first_not_pad_position(sequences, pad_token_id)
            sequences = sequences[:, trunc_pos:] # minimize left padding as much as possible to reduce subsequent costs.
            input_len -= trunc_pos
            attention_mask = sequences.not_equal(pad_token_id).to(
                dtype=torch.long, device=sequences.device)
            if eos_token_id is not None and pad_token_id == eos_token_id:
                for i in range(attention_mask.shape[0]):
                    indices = torch.where(sequences[i,:] == pad_token_id)[0]
                    if len(indices) == 0: continue
                    j = torch.searchsorted(indices, input_len)
                    if j < len(indices):
                        attention_mask[i, indices[j]] = True # include eos token if eos_token == pad_token

        if not return_action_mask:
            return sequences, attention_mask
        if eos_token_id is None:
            action_mask = torch.ones_like(sequences, dtype=torch.bool)
        else:
            # left padding may be applied, only mask action
            action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0
            action_mask = F.pad(action_mask, (1 + input_len, -1),
                                value=True)    # include eos token and input
        action_mask[:, :input_len] = False
        action_mask = action_mask[:, 1:]
        return sequences, attention_mask, action_mask[:, -(sequences.size(1) - input_len):]

    def forward(self,
                sequences: torch.LongTensor,
                num_actions: int,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns action log probs
        """
        output = self.model(sequences, attention_mask=attention_mask)
        logits = output['logits']
        log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])
        return log_probs[:, -num_actions:]
    
    @property
    def device(self):
        return self.model.device
