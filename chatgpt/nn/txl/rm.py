# encoding=utf-8
import json
import torch
import re
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
from chatgpt.nn import RewardModel
from chatgpt.nn.utils import masked_mean
from .transformerxl_modeling import GPT2Model
from transformers import T5Tokenizer, PreTrainedTokenizer


class TXLRewardModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model = GPT2Model(**config)
        reward_hidden_size = config["hidden_size"]*2
        self.reward_head = nn.Sequential(
            nn.Linear(config["hidden_size"], reward_hidden_size),
            nn.ReLU(),
            nn.Linear(reward_hidden_size, 1),
        )

    
class TransfoXLRM(RewardModel):

    def __init__(self, 
                 model: nn.Module, 
                 value_head: nn.Module,
                 rm_tokenizer:T5Tokenizer, 
                 policy_tokenizer:PreTrainedTokenizer) -> None:
        super().__init__(model, value_head)
        self.rm_tokenizer = rm_tokenizer
        self.policy_tokenizer = policy_tokenizer
        return

    def get_attn_mask(self, max_length: int, memory_length: int = 0) -> torch.Tensor:
            """ 计算Txl的注意力掩码矩阵 """
            '''memory_length默认为0时返回max_length*max_length的下三角矩阵'''
            mem_attn_msk = torch.ones(
                (max_length, max_length + memory_length), dtype=torch.long
            )

            mem_attn_msk = torch.tril(
                torch.triu(
                    mem_attn_msk, 1- max_length + memory_length
                ), memory_length
            )
            return mem_attn_msk
    
    def padding_attention_mask(self, attn_msk_list, pad_to: int):
        """
            将注意力矩阵对其相同的大小
        """

        for msk_idx in range(len(attn_msk_list)):

            attn_msk_list[msk_idx] = F.pad(
                attn_msk_list[msk_idx],
                (0, 0, 0, pad_to - attn_msk_list[msk_idx].size(0)),
                value=0
            ) # pad down
            
            attn_msk_list[msk_idx] = F.pad(
                attn_msk_list[msk_idx],
                (0, pad_to - attn_msk_list[msk_idx].size(1), 0, 0),
                value=0
            ) # pad right
        
        return torch.stack(attn_msk_list).unsqueeze(1)

    def get_inputs(self,texts):
        inputs = self.rm_tokenizer.batch_encode_plus(texts)
        input_ids = inputs['input_ids']
        item_length = [len(ids)-1 for ids in input_ids] # 无<eos>
        attn_mask = [self.get_attn_mask(len(ids)) for ids in input_ids] # 有<eos>
        input_ids = pad_sequence([torch.Tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.rm_tokenizer.pad_token_id).long()
        attn_mask = self.padding_attention_mask(attn_mask, pad_to=input_ids.shape[1])
        item_length = [index-input_ids.shape[-1]for index in item_length]
        return input_ids, item_length, attn_mask
    
    def _preprocess(self, sequences: torch.LongTensor) -> dict:

        # decode
        decode_texts = self.policy_tokenizer.batch_decode(sequences, skip_special_tokens=False)
        decode_texts = [item.replace("<|endoftext|>", "").replace("<|padding|>", "").replace("<human>", "<sep>user").replace("<bot>", "<sep>bot") for item in decode_texts]
        decode_texts = [f'<bos>{re.sub("^<sep>", "", item)}' for item in decode_texts]

        # process
        # 将\n转换为\\n，\t转换为\\t
        for i in range(len(decode_texts)):
            if "\n" in decode_texts[i] and "\\n" not in decode_texts[i]:
                decode_texts[i] = decode_texts[i].replace("\n","\\n")
            if "\t" in decode_texts[i] and "\\t" not in decode_texts[i]:
                decode_texts[i] = decode_texts[i].replace("\t","\\t")

        # encode 
        # texts = [item for sublist in decode_texts for item in sublist]
        input_ids, last_index, attn_mask = self.get_inputs(decode_texts)

        return input_ids, attn_mask, last_index
    
    def forward(self, 
                sequences: torch.LongTensor, 
                action_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        input_ids, attn_mask, last_index = self._preprocess(sequences=sequences)

        logits, last_hidden, *_ = self.body(
            input_ids.cuda(), 
            None, 
            attn_mask.cuda(), 
            mems=None, 
            extra_embeddings=None,
            return_last_hidden_state=True,
        )
        last_hidden = torch.stack([last_hidden[i,index] for i,index in enumerate(last_index)])
        output = self.value_head(last_hidden)

        return output.view(-1)


def modeling_transforxl_rm(txl_config_path:str, ckpt_path:str, txl_tokenizer_path:str, policy_tokenizer:PreTrainedTokenizer) -> TransfoXLRM:

    # load ckpt
    with open(txl_config_path, "r") as f:
        txl_config = json.load(f)
    txl_config["adapter_hidden_size"] = txl_config["adapter_hidden_size"] if "adapter_hidden_size" in txl_config else 0

    origin_rm_model = TXLRewardModel(config=txl_config)
    state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))['module']
    new_state_dict = {
        key[len("module.model."):]: value for key, value in state_dict.items()
    }
    origin_rm_model.load_state_dict(new_state_dict, strict=True)
    
    rm_tokenizer = T5Tokenizer.from_pretrained(txl_tokenizer_path)

    rm_model = TransfoXLRM(
        model=origin_rm_model.model,
        value_head=origin_rm_model.reward_head,
        rm_tokenizer=rm_tokenizer,
        policy_tokenizer=policy_tokenizer
    )

    return rm_model
