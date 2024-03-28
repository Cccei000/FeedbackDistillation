# encoding=utf-8
from typing import Callable, Optional

import datasets
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from chatgpt.utils import is_rank_0, logging_rank_0


class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int) -> None:
        super().__init__()
        self.chosen = []
        self.reject = []
        for data in tqdm(dataset, disable=not is_rank_0()):
            prompt = data['prompt']

            chosen = prompt + data['chosen'] + "<|endoftext|>"
            chosen_token = tokenizer(chosen,
                                     max_length=max_length,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            self.chosen.append({
                "input_ids": chosen_token['input_ids'],
                "attention_mask": chosen_token['attention_mask']
            })

            reject = prompt + data['rejected'] + "<|endoftext|>"
            reject_token = tokenizer(reject,
                                     max_length=max_length,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            self.reject.append({
                "input_ids": reject_token['input_ids'],
                "attention_mask": reject_token['attention_mask']
            })

    def __len__(self):
        length = len(self.chosen)
        return length

    def __getitem__(self, idx):
        return self.chosen[idx]["input_ids"], self.chosen[idx]["attention_mask"], self.reject[idx][
            "input_ids"], self.reject[idx]["attention_mask"]

class RMCollator():
    """
    Collator for reward model

    Args:
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(self, tokenizer: Callable, max_length: int, query_key:str="query", response_key:str="responses") -> None:
        self.tokenizer = tokenizer 
        self.max_seq_length = max_length
        self.query_key = query_key
        self.response_key = response_key

    def get_inputs(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt",padding=True,truncation=True,max_length=self.max_seq_length
        )
        return inputs.input_ids, inputs.attention_mask

    def __call__(self, batch):
        pairs = []
        input_ids = []
        action_masks = []
        l = 0
        tasks = []
        eos_token = self.tokenizer.eos_token # "</s>"
        for d in batch:
            query = d[self.query_key]
            query_ids = self.tokenizer(query, max_length=self.max_seq_length-1, truncation=True)["input_ids"]
            pair  = torch.combinations(torch.arange(len(d[self.response_key])), 2).tolist() if d["pairs"] is None else d["pairs"]
            pair = [[p[0]+l,p[1]+l] for p in pair]
            pairs.extend(pair)
            for i, r in enumerate(d[self.response_key]):
                # 数据中不应该有超过10个response的情况, response太多可能会OOM
                # if i > 10:
                #     print("drop responses>10")
                #     break
                if isinstance(r, str):
                    r_text = r
                else:
                    r_text = r['text'] if r['text'] is not None else ""
                response = self.tokenizer(r_text+eos_token)["input_ids"][1:]
                input_ids.append(torch.Tensor(query_ids+response))
                action_masks.append(torch.Tensor([0]*len(query_ids) + [1]*len(response)))
                l+=1
            tasks.extend([d["task"]]*len(d[self.response_key]))
        
        input_ids = [ids[:self.max_seq_length] for ids in input_ids]
        action_masks = [m[:self.max_seq_length] for m in action_masks]
        input_ids = zero_pad_sequences(input_ids, side= 'right', padding_value=self.tokenizer.pad_token_id).long()
        action_masks = zero_pad_sequences(action_masks, side= 'right', padding_value=0).long()
        attn_mask = input_ids.not_equal(self.tokenizer.pad_token_id).long()

        if self.tokenizer.eos_token_id is not None and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            last_index =  torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in attn_mask], dtype=torch.int64)
            for i,ind in enumerate(last_index):
                if ind+1<self.max_seq_length:
                    attn_mask[i,ind+1]=1
        output = {
            "input_ids":input_ids, 
            "attention_mask":attn_mask,
            "action_mask":action_masks,
            "pairs":torch.Tensor(pairs).long(),
            "task":tasks,
        }
        return output

class RMPredictCollator():
    """
    Collator for reward model

    Args:
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(self, tokenizer: Callable, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_length

    def get_inputs(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt",padding=True,truncation=True,max_length=self.max_seq_length
        )
        return inputs.input_ids, inputs.attention_mask

    def __call__(self, batch):
        prefix_user = "<human>："
        prefix_bot = "<bot>："
        texts = [prefix_user+sample['query']+prefix_bot+sample['response'] for sample in batch]  
        input_ids,attn_mask = self.get_inputs(texts)
        return {
            "input_ids":input_ids, 
            "attention_mask":attn_mask,
        }


    
def zero_pad_sequences(sequences, side: str = 'left', padding_value: int = 0) -> torch.Tensor:
    assert side in ('left', 'right')
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == 'left' else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=padding_value))
    return torch.stack(padded_sequences, dim=0)