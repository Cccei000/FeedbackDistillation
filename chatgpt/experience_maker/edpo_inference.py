# encoding=utf-8
import gc
import itertools
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from chatgpt.experience_maker import (PreferenceExperience,
                                      PreferenceExperienceMaker)
from chatgpt.nn import Actor, GSArgs
from chatgpt.nn.utils import zero_pad_sequences
from chatgpt.utils import is_rank_0, logging_rank_0


class EDPOExperienceMaker(PreferenceExperienceMaker):
    """
    EDPO experience maker.
    """

    def __init__(self,
        actor: Actor,
        initial_model: Actor = None,
        reward_model: nn.Module = None,
        seed: int = 1234,
        pad_token_id: int = 0,
        eos_token_id: int = 0,
        actor_minibatch_size: int = 16,
        rm_minibatch_size: int = 1,
        gen_minibatch_size: Optional[int] = None,
        equalizing_preferences:bool=False,
        max_n_preferences: int = 0,
        gs_args: GSArgs = None,
        gen_args: Optional[dict] = None) -> None:
        super().__init__(actor, initial_model, reward_model)
        self.actor_minibatch_size = actor_minibatch_size
        self.gen_minibatch_size = gen_minibatch_size
        self.rm_minibatch_size = rm_minibatch_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.seed = seed
        self.equalizing_preferences = equalizing_preferences
        self.max_n_preferences = max_n_preferences
        self.gs_args = gs_args
        self.device = next(self.actor.model.parameters()).device
        self.gen_args = gen_args
        torch.manual_seed(self.seed)
        return

    def make_experience_with_actor(self, inputs: List[Dict[str, torch.Tensor]], do_search: bool) -> Dict[str, torch.Tensor]:
        
        mini_batch_preference = inputs[:]
        makeup_info = {'input_ids':[], 'sample_index':[], 'num':[], 'sequences':[]}

        # 对 response 数量不足的 query 进行生成增广
        if self.equalizing_preferences: 
            for i,input in enumerate(inputs):
                cur_n = input['preference_sequences'].shape[0]
                if cur_n < self.max_n_preferences:
                    makeup_info['sample_index'].append(i)
                    makeup_info['num'].append(self.max_n_preferences - cur_n)
                    makeup_info['input_ids'].extend([input['input_ids']]*makeup_info['num'][-1])
            if len(makeup_info['input_ids']) > 0:
                input_ids = zero_pad_sequences(sequences=makeup_info['input_ids'], side='right', padding_value=self.pad_token_id)
                do_search = do_search and (self.gs_args is not None) and (self.gs_args.enabling_tot)
                with torch.no_grad():
                    # 生成文本
                    gen_batch_size = self.gen_minibatch_size if self.gen_minibatch_size else self.actor_minibatch_size
                    if do_search:
                        assert self.reward_model is not None,  "a reward model expected, got None"
                        self.reward_model = self.reward_model.cuda()
                    for i in tqdm(range(0, len(input_ids), gen_batch_size),
                            desc=f"Generating",
                            disable=not is_rank_0()):
                        mini_batch_input = input_ids[i: i + gen_batch_size]
                        outputs = self.actor.module.generate(
                            mini_batch_input,
                            gs_args=self.gs_args,
                            reward_model=self.reward_model,
                            **self.gen_args
                        )
                        sequences = outputs[0]
                        gc.collect()
                        makeup_info['sequences'].append(sequences)
                
                    if do_search:
                        self.reward_model = self.reward_model.cpu()

                max_len = max(item.shape[1] for item in makeup_info['sequences']) 
                makeup_sequences = torch.cat([
                            F.pad(item, (0, max_len - item.shape[1]), value=self.pad_token_id) # 使用传入的pad_token_id填充
                            for item in makeup_info['sequences']
                        ], dim=0)
                end_idx = list(itertools.accumulate(makeup_info['num']))
                start_idx =  [0] + end_idx[:-1]
 
                for i,start,end in zip(makeup_info['sample_index'], start_idx, end_idx):
                    preference_seqs = [row for row in inputs[i]['preference_sequences']] \
                                    + [row for row in makeup_sequences[start:end]]
                    preference_seqs = [item.cuda() for item in preference_seqs]
                    mini_batch_preference[i]['preference_sequences'] = zero_pad_sequences(sequences=preference_seqs, side='right', padding_value=self.pad_token_id)
    
 
        # construct action_mask and attention_mask
        mini_batch_response = defaultdict(list)
        for i,input in enumerate(mini_batch_preference):
            attention_mask = self.actor.module._create_attn_mask(input['preference_sequences'], \
                                    eos_token_id=self.eos_token_id, pad_token_id=self.pad_token_id)
            pref_bz = input['preference_sequences'].shape[0] 
            action_mask = self.actor.module._create_action_mask(input['input_ids'].repeat(pref_bz).view(pref_bz, -1), \
                            input['preference_sequences'], eos_token_id=self.eos_token_id, pad_token_id=self.pad_token_id)
            mini_batch_preference[i]['attention_mask'] = attention_mask
            mini_batch_preference[i]['action_mask'] = action_mask[:, 1:].to(self.device)
            
            mini_batch_response['sequence'].append(input['preference_sequences'])
            mini_batch_response['attention_mask'].append(mini_batch_preference[i]['attention_mask'])
            mini_batch_response['action_mask'].append(mini_batch_preference[i]['action_mask'])
            mini_batch_response['n_preference'].append(pref_bz)

        # 对齐生成的文本（统一右填充）
        for key, value in mini_batch_response.items():
            if key == 'n_preference': continue
            max_len = max(item.shape[1] for item in value)
            if key == "sequence":
                padded_value = [
                    F.pad(item, (0, max_len - item.shape[1]), value=self.pad_token_id) # 使用传入的pad_token_id填充
                    for item in value
                ]
            else:
                padded_value = [
                    F.pad(item, (0, max_len - item.shape[1]), value=0)
                    for item in value
                ]
            mini_batch_response[key] = torch.cat(padded_value, dim=0) # (num, seq_len)

        
        # print_rank_0("mini_batch_response['sequence']", mini_batch_response['sequence'])
            # actor 推理
        for i in tqdm(range(0, mini_batch_response['sequence'].shape[0], self.actor_minibatch_size),
                        desc=f"Infer actor model",
                        disable=not is_rank_0()): 
            start, end = i, i + self.actor_minibatch_size
            sequences = mini_batch_response["sequence"][start : end]                # (bs, seq_len)
            attention_mask = mini_batch_response["attention_mask"][start : end]
            
            num_actions = mini_batch_response["action_mask"].shape[1]
            action_mask = mini_batch_response["action_mask"][start : end]
            # action_log_probs = self.actor(sequences.to(self.device), num_actions, attention_mask.to(self.device))   # (bs, seq_len - 1)


            # mini_batch_response["action_log_probs"].append(action_log_probs)

        
        # mini_batch_response["action_log_probs"] = torch.cat(mini_batch_response["action_log_probs"], dim=0) #(bs*n_pref, seq_len-1)

        end_idx = list(itertools.accumulate(mini_batch_response['n_preference']))
        start_idx = [0] + end_idx[:-1]

        # print_rank_0(f'start_idx {start_idx}, end_idx {end_idx}')

        for i,(start,end) in enumerate(zip(start_idx, end_idx)):
            mini_batch_preference[i]['preference_sequences'] = mini_batch_response['sequence'][start:end]
            mini_batch_preference[i]['action_mask'] = mini_batch_response['action_mask'][start:end]
            mini_batch_preference[i]['attention_mask'] = mini_batch_response['attention_mask'][start:end]
            # mini_batch_preference[i]['action_log_probs'] = mini_batch_response['action_log_probs'][start:end]
            # print_rank_0(f"mini_batch_preference {i}: \
            #     preference {mini_batch_preference[i]['preference_sequences'].shape}\
            #     action_mask {mini_batch_preference[i]['action_mask'].shape}\
            #     attention_mask {mini_batch_preference[i]['attention_mask'].shape}\
            #     action_log_probs {mini_batch_preference[i]['action_log_probs'].shape}")


        return mini_batch_response, mini_batch_preference, makeup_info['sample_index']
    
    
    def make_experience_with_initial_model(self, mini_batch_preference:List[Dict[str, torch.Tensor]], mini_batch_response:Dict[str,torch.Tensor]) -> torch.Tensor:

        if self.initial_model is not None:
            self.initial_model = self.initial_model.cuda()

        total_num = mini_batch_response["sequence"].shape[0]
        batch_base_action_log_probs = []

        with torch.no_grad():
            for i in tqdm(range(0, total_num, self.rm_minibatch_size), 
                          desc=f"Infer initial model",
                          disable=not is_rank_0()):
                start, end = i, i + self.rm_minibatch_size
                sequences = mini_batch_response["sequence"][start : end]
                attention_mask = mini_batch_response["attention_mask"][start : end]
                num_actions = mini_batch_response["action_mask"].shape[1]
                if self.initial_model is not None:
                    base_action_log_probs = self.initial_model(sequences.to(self.device), num_actions, attention_mask.to(self.device))  # (bs, seq_len - 1)
                else:
                    base_action_log_probs = torch.zeros_like(mini_batch_response["action_mask"][start:end])
                batch_base_action_log_probs.append(base_action_log_probs)

        if self.initial_model is not None:
            self.initial_model = self.initial_model.cpu()
        torch.cuda.empty_cache()
        gc.collect()

        # 因为输入已经padding过了，此时base_action_log_probs是已经对齐了的
        batch_base_action_log_probs = torch.cat(batch_base_action_log_probs, dim=0)

        end_idx = list(itertools.accumulate(mini_batch_response['n_preference']))
        start_idx = [0] + end_idx[:-1]

        for i,(start,end) in enumerate(zip(start_idx, end_idx)):
            mini_batch_preference[i]['ref_action_log_probs'] = batch_base_action_log_probs[start:end]
        mini_batch_response["ref_action_log_probs"] = batch_base_action_log_probs 

        return mini_batch_response, mini_batch_preference

    def make_experience_with_reward_model(self, mini_batch_preference:Dict[str,torch.Tensor], \
                                        mini_batch_response:List[Dict[str, torch.Tensor]], \
                                        makeup_list:List) -> List[PreferenceExperience]:
        if len(makeup_list) > 0:
            self.reward_model.cuda()
        
        start_idx = itertools.accumulate([0] + mini_batch_response['n_preference'])
        reordered_row_idx = []
        [reordered_row_idx.extend(list(range(start_idx[i], start_idx[i+1]))) for i in makeup_list]
        total_num = len(reordered_row_idx)
        
        exps = []
        rewards = []
        with torch.no_grad():
            for i in tqdm(range(0, total_num, self.rm_minibatch_size), 
                          desc=f"Scoring",
                          disable=not is_rank_0()):

                start, end = i, i + self.rm_minibatch_size
                sequences = mini_batch_response["sequence"][reordered_row_idx[start : end]]                            # (bs, seq_len)
                attention_mask = mini_batch_response["attention_mask"][reordered_row_idx[start : end]]                 # (bs, seq_len)
                action_mask = mini_batch_response["action_mask"][reordered_row_idx[start : end]] 
                r = self.reward_model(sequences.to(self.device), action_mask.to(self.device), attention_mask.to(self.device)) # (bs,)
                rewards.extend(r.detach().cpu().tolist())
        
            # reorder preference_sequences according to the reward
            acc_p = 0
            for i,input in enumerate(mini_batch_preference):
                if i in makeup_list:
                    n_pref = mini_batch_response['n_preference'][i] 
                    assert input['preference_sequences'].shape[0] == n_pref 

                    scores = rewards[acc_p+i: acc_p+i+n_pref]
                    acc_p += n_pref
                    sorted_indices = torch.argsort(torch.tensor(scores))
                    preference_sequences = input['preference_sequences'][sorted_indices]
                    # action_log_probs = input['action_log_probs'][sorted_indices]
                    ref_action_log_probs = input['ref_action_log_probs'][sorted_indices]
                    attention_mask = input['attention_mask'][sorted_indices]
                    action_mask = input['action_mask'][sorted_indices]
                else:
                    preference_sequences = input['preference_sequences']
                    # action_log_probs = input['action_log_probs'] # (bs, seq_len - 1)
                    ref_action_log_probs = input['ref_action_log_probs'] # (bs, seq_len - 1)
                    attention_mask = input['attention_mask'] 
                    action_mask = input['action_mask'] # (bs, seq_len - 1)
                
                task = input['task'] if 'task' in input.keys() else 'default'
                # exps.append(PreferenceExperience(preference_sequences, action_log_probs, ref_action_log_probs,
                #             attention_mask, action_mask))
                exps.append(PreferenceExperience(task, preference_sequences, ref_action_log_probs,
                            attention_mask, action_mask))
        
        if len(makeup_list) > 0:
            self.reward_model.cpu()

        return exps

    @torch.no_grad()
    def make_experience(self, inputs: List[Dict[str, torch.Tensor]], return_kl:bool=False, do_search=False) -> List[PreferenceExperience]:

        self.actor.eval()

        # actor forward
        mini_batch_response, mini_batch_preference, makeup_list = self.make_experience_with_actor(inputs=inputs, do_search=do_search)
        gc.collect()
        # init_model forward
        if  self.initial_model is not None:
            self.initial_model.eval().cpu()
        mini_batch_response, mini_batch_preference = self.make_experience_with_initial_model(mini_batch_preference, mini_batch_response)
        gc.collect()
        # 再做rw forward

        if self.reward_model is not None:
            self.reward_model.eval().cpu()

        exps = self.make_experience_with_reward_model(mini_batch_preference, mini_batch_response, makeup_list)
        gc.collect()
        
        # if return_kl:
        #     approx_kl = compute_approx_kl(
        #         mini_batch_response["action_log_probs"], mini_batch_response["ref_action_log_probs"], action_mask=mini_batch_response["action_mask"], return_mean=True
        #     )

        #     return exps, approx_kl.mean()
            
        return exps