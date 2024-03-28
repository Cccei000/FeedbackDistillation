from typing import Optional, Tuple, Union, Callable, Dict
import gc
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM
from chatgpt.nn import Reflector


class LlamaReflector(Reflector):
    """
    Llama reflect model.

    """

    def __init__(self, model: LlamaForCausalLM) -> None:
        super().__init__(model)

    def generate(self, input_ids: torch.Tensor, **kwargs):
        pad_token_id = kwargs.get('pad_token_id', None)
        eos_token_id = kwargs.get('eos_token_id', None)
        assert pad_token_id is not None and eos_token_id is not None, "pad_token_id and eos_token_id must be provided."

        output = self.model.generate(input_ids, **kwargs)
        if isinstance(output, tuple):
            sequences, gen_info = output
        else:
            sequences, gen_info = output, []

        if pad_token_id != eos_token_id: # take eos in the middle into account
            attention_mask = sequences != pad_token_id
            pad_mask = (input_ids == pad_token_id).to(device=sequences.device)
        else: # does not take eos in the middle into account, too complicate
            attention_mask = (sequences == eos_token_id).cumsum(dim=-1) <= 1
            pad_mask = (input_ids == pad_token_id).to(device=sequences.device)
        action_mask = attention_mask.clone()
        action_mask[:, :pad_mask.shape[1]] *= pad_mask
        action_mask = action_mask.bool()

        # attention_mask = None
        # if pad_token_id is not None:
        #     if eos_token_id is not None and eos_token_id == pad_token_id:
        #         attention_mask = (sequences == pad_token_id).cumsum(dim=-1) <= 1
        #     else:
        #         attention_mask = sequences.not_equal(pad_token_id)
        #     attention_mask.to(dtype=torch.long, device=sequences.device)

        # input_mask = (input_ids == pad_token_id).cumsum(dim=-1) != 0            # 标记input_ids中的pad token
        # input_mask = input_mask.to(sequences.device)
        # if eos_token_id is None:
        #     action_mask = torch.ones_like(sequences, dtype=torch.bool)          # (bs, seq_len)
        # else:          
        #     action_mask = (sequences == eos_token_id).cumsum(dim=-1) == 0       # (bs, seq_len)
        #     action_mask = F.pad(action_mask, (1, -1), value=True)               # (bs, seq_len)
        # action_mask[:, :input_mask.shape[1]] *= input_mask

        return sequences, attention_mask, action_mask[:, 1:], gen_info

    