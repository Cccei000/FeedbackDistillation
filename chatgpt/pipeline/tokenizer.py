# encoding=utf-8
from typing import Union, List, Dict
from collections import defaultdict

import torch
import torch.nn.functional as F
from tokenizers import AddedToken
from transformers import AutoTokenizer, LlamaTokenizer

from chatgpt.nn.utils import zero_pad_sequences
from chatgpt.pipeline.config import PPOPipelineConfig
from chatgpt.utils import logging_rank_0
from chatgpt.models.baichuan import BaichuanTokenizer
import chatgpt.pipeline.feedback_distill_template as template

### Build Tokenizer ###

def bulid_llama_13B_tokenizer(tokenizer_path:str) -> AutoTokenizer:
    special_token_dict = {'pad_token': '</s>'}
    human = AddedToken("<human>",lstrip=False,rstrip=False,single_word=False,normalized=True)
    bot = AddedToken("<bot>",lstrip=False,rstrip=False,single_word=False,normalized=True)
    llama_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    llama_tokenizer.add_special_tokens(special_token_dict)
    llama_tokenizer.add_special_tokens({"additional_special_tokens":[human, bot]})
    return llama_tokenizer


def build_llama2_13B_tokenizer(tokenizer_path:str) -> AutoTokenizer:
    special_token_dict = {'pad_token': '</s>'}
    llama_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    llama_tokenizer.add_special_tokens(special_token_dict)
    return llama_tokenizer


def build_baichuan_13B_tokenizer(tokenizer_path:str) -> AutoTokenizer:
    llama_tokenizer = BaichuanTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    return llama_tokenizer


def build_other_tokenizer(tokenizer_path:str) -> AutoTokenizer:
    llama_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    return llama_tokenizer

TOKENIZER = defaultdict(lambda: build_other_tokenizer)
TOKENIZER["llama_13B"] = bulid_llama_13B_tokenizer
TOKENIZER["llama2_13B"] = build_llama2_13B_tokenizer
TOKENIZER["baichuan_13B"] = build_baichuan_13B_tokenizer

### align different prompts between policy and rm ###

def llama2_to_llama(args:PPOPipelineConfig, src_tokenizer:Union[LlamaTokenizer,AutoTokenizer], dst_tokenizer:Union[LlamaTokenizer,AutoTokenizer]):
    
    import re
    
    def process_text(text:str) -> str:
        """对话角色prompt转换

        Args:
            text (str): _description_

        Returns:
            str: _description_
        """
        text = text.replace("<Human Round-1>:", "<human>:")
        text = re.sub(r"[\s]*<Human Round-[\d]+>:", "\n<human>:", text)
        text = re.sub(r"\n<Assistant Round-[\d]+>:", "\n<bot>:", text)
        
        return text
        
    def convert_func(sequences:torch.Tensor, attention_mask:torch.Tensor, action_mask:torch.Tensor, device:torch.device):
        
        if action_mask is None:
            action_mask = attention_mask[:, 1:]
        
        assert sequences.shape[0] == attention_mask.shape[0] and sequences.shape[0] == action_mask.shape[0]
        assert attention_mask.ndim == 2 and action_mask.ndim == 2
        assert attention_mask.shape[1] == action_mask.shape[1] + 1
        token_ids = []
        attn_msk_list = []
        act_msk_list = []
        input_mask = attention_mask ^ F.pad(action_mask, (1,0), value=False)
        input_last_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in input_mask], dtype=torch.int64, device=device)
        seq_last_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in attention_mask], dtype=torch.int64, device=device)

        # logged=False

        for seq, attn_msk, act_msk, idx, eos_idx in zip(sequences, attention_mask, action_mask, input_last_index, seq_last_index):
            seq_text = src_tokenizer.decode(seq, skip_special_tokens=True)
            input_text = src_tokenizer.decode(seq[:idx+1], skip_special_tokens=True)
  
            seq_text = process_text(seq_text)
            input_text = process_text(input_text)

            seq_dst = dst_tokenizer(seq_text, add_special_tokens=True)
            input_dst = dst_tokenizer(input_text, add_special_tokens=True)
            
            # 手动加上eos id
            if seq[eos_idx] == src_tokenizer.eos_token_id:
                seq_dst["input_ids"] += [dst_tokenizer.eos_token_id]
                seq_dst["attention_mask"] += [1]

            seq_len = min(len(seq_dst["input_ids"]), args.rm_max_seq_len)
            input_len = len(input_dst["input_ids"])
            act_msk_dst = torch.ones(seq_len, device=device, dtype=action_mask.dtype)
            act_msk_dst[:input_len] = False
               
            if len(seq_dst["input_ids"]) > args.rm_max_seq_len:
                print(f"Truncate respond: {len(seq_dst['input_ids'])} -> {args.rm_max_seq_len}")
                
            # if not logged:
            #     logging_rank_0(f"Text: {seq_text!r}")
            #     logging_rank_0(f"Ids: {seq_dst['input_ids']}")
            #     logging_rank_0(f"AttnMsk: {seq_dst['attention_mask']}")
            #     logging_rank_0(f"ActMsk: {act_msk_dst[1:]}")
            #     logged = True

            token_ids.append(torch.tensor(seq_dst["input_ids"], dtype=sequences.dtype, device=device)[:seq_len])
            attn_msk_list.append(torch.tensor(seq_dst["attention_mask"], dtype=attention_mask.dtype, device=device)[:seq_len])
            act_msk_list.append(act_msk_dst[1:])

        # print_rank_0(f'sequence_texts {sequence_texts} query_texts {query_texts}')
        token_ids = zero_pad_sequences(token_ids, "right", padding_value=dst_tokenizer.pad_token_id)
        attn_msk = zero_pad_sequences(attn_msk_list, "right", padding_value=False)
        act_msk = zero_pad_sequences(act_msk_list, "right", padding_value=False)
        
        return token_ids, attn_msk, act_msk
    
    return convert_func


def baichuan_to_llama2(args:PPOPipelineConfig, src_tokenizer:BaichuanTokenizer, dst_tokenizer:Union[LlamaTokenizer,AutoTokenizer]):
    
    import re
    
    def process_text(text:str) -> str:
        """对话角色prompt转换

        Args:
            text (str): _description_

        Returns:
            str: _description_
        """
        turns = text.split("<reserved_106>")[1:]
        output = ""
        counter = 1
        for turn in turns:
            texts = turn.split("<reserved_107>")
            output += f"<Human Round-{counter}>:{texts[0]}<Assistant Round-{counter}>{texts[1]}"
            counter += 1
            
        return output
        
    def convert_func(sequences:torch.Tensor, attention_mask:torch.Tensor, action_mask:torch.Tensor, device:torch.device):
        
        if action_mask is None:
            action_mask = attention_mask[:, 1:]
        
        assert sequences.shape[0] == attention_mask.shape[0] and sequences.shape[0] == action_mask.shape[0]
        assert attention_mask.ndim == 2 and action_mask.ndim == 2
        assert attention_mask.shape[1] == action_mask.shape[1] + 1
        token_ids = []
        attn_msk_list = []
        act_msk_list = []
        input_mask = attention_mask ^ F.pad(action_mask, (1,0), value=False)
        input_last_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in input_mask], dtype=torch.int64, device=device)
        seq_last_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in attention_mask], dtype=torch.int64, device=device)

        # logged=False

        for seq, attn_msk, act_msk, idx, eos_idx in zip(sequences, attention_mask, action_mask, input_last_index, seq_last_index):
            seq_text = src_tokenizer.decode(seq, skip_special_tokens=True)
            input_text = src_tokenizer.decode(seq[:idx+1], skip_special_tokens=True)
  
            seq_text = process_text(seq_text)
            input_text = process_text(input_text)

            seq_dst = dst_tokenizer(seq_text, add_special_tokens=True)
            input_dst = dst_tokenizer(input_text, add_special_tokens=True)
            
            # 手动加上eos id
            if seq[eos_idx] == src_tokenizer.eos_token_id:
                seq_dst["input_ids"] += [dst_tokenizer.eos_token_id]
                seq_dst["attention_mask"] += [1]

            seq_len = min(len(seq_dst["input_ids"]), args.rm_max_seq_len)
            input_len = len(input_dst["input_ids"])
            act_msk_dst = torch.ones(seq_len, device=device, dtype=action_mask.dtype)
            act_msk_dst[:input_len] = False
               
            if len(seq_dst["input_ids"]) > args.rm_max_seq_len:
                print(f"Truncate respond: {len(seq_dst['input_ids'])} -> {args.rm_max_seq_len}")
                
            # if not logged:
            #     logging_rank_0(f"Text: {seq_text!r}")
            #     logging_rank_0(f"Ids: {seq_dst['input_ids']}")
            #     logging_rank_0(f"AttnMsk: {seq_dst['attention_mask']}")
            #     logging_rank_0(f"ActMsk: {act_msk_dst[1:]}")
            #     logged = True

            token_ids.append(torch.tensor(seq_dst["input_ids"], dtype=sequences.dtype, device=device)[:seq_len])
            attn_msk_list.append(torch.tensor(seq_dst["attention_mask"], dtype=attention_mask.dtype, device=device)[:seq_len])
            act_msk_list.append(act_msk_dst[1:])

        # print_rank_0(f'sequence_texts {sequence_texts} query_texts {query_texts}')
        token_ids = zero_pad_sequences(token_ids, "right", padding_value=dst_tokenizer.pad_token_id)
        attn_msk = zero_pad_sequences(attn_msk_list, "right", padding_value=False)
        act_msk = zero_pad_sequences(act_msk_list, "right", padding_value=False)
        
        return token_ids, attn_msk, act_msk
    
    return convert_func


GLUE = {
    "llama2_13B_to_llama_13B":  llama2_to_llama,
    "baichuan_13B_to_llama2_13B": baichuan_to_llama2,
}


### prompt convertion between actor and reflector ###

class FDPromptConvertion:

    def __init__(self, 
        actor_tokenizer: Union[LlamaTokenizer, AutoTokenizer],
        reflector_tokenizer: Union[LlamaTokenizer, AutoTokenizer],
        reflector_type: str) -> None:

        self.actor_tokenizer = actor_tokenizer
        self.reflector_tokenizer = reflector_tokenizer
        self.reflector_type = reflector_type
        self.actor_template = template.actor_template
        self.reflector_template = template.reflector_template[self.reflector_type]
        self.refinement_template = template.refinement_template

        assert '{__query__}' in self.actor_template, \
            "Actor template must contain format fields '{__query__}'."
        assert '{__query__}' in self.reflector_template and '{__response__}' in self.reflector_template, \
            "Reflector template must contain format fields '{__query__}' and '{__response__}'."
        assert '{__query__}' in self.refinement_template and '{__response__}' in self.refinement_template \
            and '{__feedback__}' in self.refinement_template, \
            "Refinement template must contain format fields '{__query__}', '{__response__}', and '{__feedback__}'."
        

    def to_reflector(self, queries: List[str], sequences: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:  

        action_mask = F.pad(action_mask, (1, 0), value=False)
        response_text = self.actor_tokenizer.batch_decode(
            sequences * action_mask,
            skip_special_tokens=True
        )
        query_text = queries
        input_text = [self.reflector_template.format(__query__=q.strip(), __response__=r.strip()) 
                      for (q, r) in zip(query_text, response_text)]
        input_ids = self.reflector_tokenizer(input_text, return_tensors='pt', padding=True).input_ids    

        return input_ids


    def to_actor(self, queries: List[str]) -> torch.Tensor:
        queries = [self.actor_template.format(__query__=q) for q in queries]
        input_ids = self.actor_tokenizer(queries, return_tensors='pt', padding=True).input_ids
        return input_ids


    def refine(self, queries: List[str], response_dict: Dict[str, torch.Tensor], feedback_dict: Dict[str, torch.Tensor], pad_token_id: int):

        assert len(queries) == len(response_dict['sequences']) and len(queries) == len(feedback_dict['sequences'])

        feedback_mask = F.pad(feedback_dict['action_mask'], (1, 0), value=False)
        feedback_text = self.reflector_tokenizer.batch_decode(
            feedback_dict['sequences'] * feedback_mask,
            skip_special_tokens=True
        )
        feedback_text = [f.strip() for f in feedback_text]


        # this is hard-coded for Auto-J or ultraCM
        feedback_score = [] 
        for f in feedback_text:
            try:
                if self.reflector_type == 'auto-j_13B':
                    feedback_score.append(int(f.split('[[')[1].split(']]')[0]))
                elif self.reflector_type == 'ultraCM_13B':
                    score = []
                    for each in f:
                        if each.isdigit():
                            score.append(each)
                        else:
                            break
                    feedback_score.append(int(''.join(score)))
                else:
                    raise Exception
            except:
                feedback_score.append(-1)
        repeation_score = []
        for f in feedback_text:
            repeation_score.append(int(any([w in f for w in ['repeat', 'repeti']])))


        response_mask = F.pad(response_dict['action_mask'], (1, 0), value=False)
        response_text = self.actor_tokenizer.batch_decode(
            response_dict['sequences'] * response_mask,
            skip_special_tokens=True
        )
        response_text = [r.strip() for r in response_text]
        query_text = queries

        return_dict = defaultdict(list)

        for i in range(len(queries)):
            input_text = self.refinement_template.format(
                __query__=query_text[i],
                __response__=response_text[i],
                __feedback__=feedback_text[i]
            )
            response_ids = torch.masked_select(response_dict['sequences'][i], response_mask[i]).reshape(1, -1)
            input_ids = self.actor_tokenizer(input_text, return_tensors='pt').input_ids.reshape(1, -1)
            sequence = torch.cat([input_ids.to(response_ids.device), response_ids], dim=-1)
            attention_mask = torch.ones_like(sequence, dtype=torch.bool)
            action_mask = torch.ones_like(sequence, dtype=torch.bool)
            action_mask[:, :input_ids.shape[-1]] = False
            return_dict['sequences'].append(sequence)
            return_dict['attention_mask'].append(attention_mask)
            return_dict['action_mask'].append(action_mask[:, 1:])

        return_dict['sequences'] = zero_pad_sequences(return_dict['sequences'], 'right', pad_token_id)
        return_dict['attention_mask'] = zero_pad_sequences(return_dict['attention_mask'], 'right', False)
        return_dict['action_mask'] = zero_pad_sequences(return_dict['action_mask'], 'right', False)
        
        return return_dict, feedback_score, repeation_score
