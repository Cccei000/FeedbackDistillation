# encoding=utf-8
from chatgpt.nn import Critic
from typing import Optional
import torch
from chatgpt.nn.utils import masked_mean
from chatgpt.nn.chatglm import load_chatglm_lm_ckpt, get_position_ids, get_masks


class ChatGLMCritic(Critic):
    """
    GPT-NeoX Actor model.

    Args:
        model: Pretrained model
    """

    def __init__(self, ckpt_path:str, bos_token_id:int, pad_token_id:int, gmask_token_id:int) -> None:
        super().__init__(load_chatglm_lm_ckpt(ckpt_path))
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.gmask_token_id = gmask_token_id
        return
        
    def forward(self,
                sequences: torch.LongTensor,
                action_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        outputs = self.model(
            sequences,
            attention_mask=get_masks(input_ids=sequences, bos_token_id=self.bos_token_id, pad_token_id=self.pad_token_id, device=self.model.device),
            position_ids=get_position_ids(input_ids=sequences, gmask_token_id=self.gmask_token_id, bos_token_id=self.bos_token_id, pad_token_ids=self.pad_token_id, device=self.model.device),
            return_dict=True
        )
        last_hidden_states = outputs.last_hidden_state.permute(1, 0, 2)

        values = self.value_head(last_hidden_states).squeeze(-1)[:, :-1]   # 这里要错位，取state的隐层 # (bs, seq_len)

        if action_mask is not None:
            num_actions = action_mask.size(1)
            values = values[:, -num_actions:]   # (bs, num_act)
            values = masked_mean(values, action_mask, dim=1) # (bs,)
            return values
        values = masked_mean(values, attention_mask[:,:-1], dim=1)  # (bs,)
        return values
    