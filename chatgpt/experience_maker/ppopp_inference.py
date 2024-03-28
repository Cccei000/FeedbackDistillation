import gc
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.profiling.flops_profiler import FlopsProfiler
from tqdm import tqdm

from chatgpt.experience_maker import Experience, LocalInferExperienceMaker
from chatgpt.nn import Actor
from chatgpt.nn.utils import compute_approx_kl
from chatgpt.utils import (ACTOR_INFER, CRITIC_INFER, GENERATE, INFER_MODELS,
                           REF, CostTimer, DeepspeedFlopsTimerGroup,
                           FlopsTimerGroup, is_rank_0, logging_rank_0)

# from chatgpt.trainer import get_rank


class PPOPPExperienceMaker(LocalInferExperienceMaker):
    """
    Naive experience maker.
    """

    def __init__(self, use_guide_action: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.use_guide_action = use_guide_action
        return

    def make_experience_with_actor_critic(self, input_ids: torch.Tensor, pure_input_ids:Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Actor、Critic推理：
            先使用Actor生成，再使用Actor和Critic推理得到logprobs和values

        Args:
            input_ids (torch.Tensor): 输入Query的token id，其中包含guidance的回答
            pure_input_ids (Optional[torch.Tensor], optional): 输入Query的token id，其中不包含guidance的回答. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: 推理结果
        """
        mini_batch_dict, mini_batch_response = self._generate_with_actor(input_ids=input_ids, pure_input_ids=pure_input_ids)
        torch.cuda.empty_cache()
        gc.collect()
        
        mini_batch_response = self._forward_with_actor_critic(input_ids=input_ids, mini_batch_dict=mini_batch_dict, mini_batch_response=mini_batch_response)
        torch.cuda.empty_cache()
        gc.collect()

        return mini_batch_response
    
    def _generate_with_actor(self, input_ids: torch.Tensor, pure_input_ids:Optional[torch.Tensor]=None):
        """Actor 生成

        Args:
            input_ids (torch.Tensor): _description_
            pure_input_ids (Optional[torch.Tensor], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        mini_batch_dict = defaultdict(list)
        mini_batch_response = {}
        
        with torch.no_grad():
            # 生成文本
            gen_batch_size = self.gen_minibatch_size if self.gen_minibatch_size else self.actor_minibatch_size
            for i in tqdm(range(0, len(input_ids), gen_batch_size),
                          desc=f"Generating",
                          disable=not is_rank_0()):
                mini_batch_input = input_ids[i: i + gen_batch_size]
                
                self._on_forward_start(GENERATE)
                sequences, attention_mask, action_mask, gen_info = self.actor.module.generate(mini_batch_input,**self.gen_args)
                self._on_forward_end(model_name=GENERATE, time=gen_info["total_time"], batch_info=gen_info["total_tokens"])
                torch.cuda.empty_cache()
                gc.collect()
                
                # Hack action mask, copied from actor.py
                if pure_input_ids is not None and self.use_guide_action:
                    mini_batch_pure_input = pure_input_ids[i: i + gen_batch_size]
                    pad_token_id = self.pad_token_id
                    eos_token_id = self.eos_token_id
                    
                    input_mask = (mini_batch_pure_input == pad_token_id).cumsum(dim=-1) != 0 # 标记input_ids中的pad token
                    input_mask = input_mask.to(self.actor.model.device)
                    
                    if eos_token_id is None:
                        mini_batch_action_mask = torch.ones_like(sequences, dtype=torch.bool)
                    else:
                        mini_batch_action_mask = (sequences == eos_token_id).cumsum(dim=-1) == 0
                        mini_batch_action_mask = F.pad(mini_batch_action_mask, (1, -1), value=True)
                    
                    mini_batch_action_mask = mini_batch_action_mask.to(self.actor.model.device)
                    mini_batch_action_mask[:, :input_mask.shape[1]] *= input_mask
                    action_mask = mini_batch_action_mask[:, 1:]
                    
                
                mini_batch_dict["sequence"].append(sequences)               # (bs, seq_len)
                mini_batch_dict["attention_mask"].append(attention_mask)    # (bs, seq_len)
                mini_batch_dict["action_mask"].append(action_mask)          # (bs, seq_len)
            
            logging_rank_0(f"Padding", "debug")
            
            # 对齐生成的文本（统一右填充）
            for key, value in mini_batch_dict.items():
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
        
        return mini_batch_dict, mini_batch_response
    
    def _forward_with_actor_critic(self, input_ids, mini_batch_response, mini_batch_dict) -> torch.Tensor:
        """Actor和Critic进行前向

        Args:
            input_ids (_type_): _description_
            mini_batch_response (_type_): _description_
            mini_batch_dict (_type_): _description_

        Returns:
            torch.Tensor: _description_
        """        
        # actor 和 critic 推理
        for i in tqdm(range(0, len(input_ids), self.actor_minibatch_size),
                        desc=f"Infer actor & critic model",
                        disable=not is_rank_0()): 
            start, end = i, i + self.actor_minibatch_size
            sequences = mini_batch_response["sequence"][start : end]                # (bs, seq_len)
            attention_mask = mini_batch_response["attention_mask"][start : end]
            num_actions = mini_batch_response["action_mask"].shape[1]
            action_mask = mini_batch_response["action_mask"][start : end]
            
            self._on_forward_start(ACTOR_INFER)
            action_log_probs = self.actor(sequences, num_actions, attention_mask)   # (bs, seq_len - 1)
            self._on_forward_end(ACTOR_INFER, batch_size=sequences.shape[0], seq_length=sequences.shape[1], time=CostTimer.get_time())
            
            self._on_forward_start(CRITIC_INFER)
            value = self.critic(sequences, action_mask, attention_mask)             # if enable_gae: (bs, seq_len - 1) else: (bs,)
            self._on_forward_end(CRITIC_INFER, batch_size=sequences.shape[0], seq_length=sequences.shape[1], time=CostTimer.get_time())

            mini_batch_dict["action_log_probs"].append(action_log_probs)
            mini_batch_dict["value"].append(value)
    
        mini_batch_response["action_log_probs"] = torch.cat(mini_batch_dict["action_log_probs"], dim=0)
        mini_batch_response["value"] = torch.cat(mini_batch_dict["value"], dim=0)  #(num,)
        
        return mini_batch_response
    
    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor, pure_input_ids:Optional[torch.Tensor]=None, return_kl:bool=False, step:int=0) -> List[Experience]:
        """采样经验池

        Args:
            input_ids (torch.Tensor): 输入Query的token id，若pure_input_ids不为None，则其中包含guidance的回答
            pure_input_ids (Optional[torch.Tensor], optional): 输入Query的token id，其中不包含guidance的回答. Defaults to None.
            return_kl (bool, optional): 是否返回经验池样本与SFT模型的approx kl信息. Defaults to False.
            step (int, optional): global step. Defaults to 0.

        Returns:
            List[Experience]: _description_
        """        
        # FIXME: 使用DeepspeedHybridEngine，调用eval会报错
        try:
            self.actor.eval()
            self.critic.eval()
        except:
            self.actor.module.train(False)
            self.critic.module.train(False)
        self.initial_model.eval()#.cpu()
        self.reward_model.eval()#.cpu()
        
        if not self.enable_policy_lora:
            self.initial_model.cpu()
        if not self.enable_rm_lora:
            self.reward_model.cpu()

        # 先做actor critic forward
        mini_batch_response = self.make_experience_with_actor_critic(input_ids=input_ids, pure_input_ids=pure_input_ids)
        torch.cuda.empty_cache()
        gc.collect()
        
        # 再做init_model forward
        batch_base_action_log_probs = self.make_experience_with_initial_model(mini_batch_response)
        mini_batch_response["base_action_log_probs"] = batch_base_action_log_probs
        torch.cuda.empty_cache()
        gc.collect()
        # 再做rw forward
        exps = self.make_experience_with_reward_model(mini_batch_response)
        torch.cuda.empty_cache()
        gc.collect()
        
        self._on_profile_flops(step=step, model_name=INFER_MODELS)
        
        if return_kl:
            approx_kl = compute_approx_kl(
                mini_batch_response["action_log_probs"], mini_batch_response["base_action_log_probs"], action_mask=mini_batch_response["action_mask"], return_mean=True
            )
            # print(f"KL:{approx_kl.mean().item()}")
            # if approx_kl.mean().item() > 1:
            #     print(f'ERROR: {mini_batch_response["sequence"].tolist()} | {mini_batch_response["action_log_probs"].tolist()} | {mini_batch_response["base_action_log_probs"].tolist()}')

            return exps, approx_kl.mean()
            
        return exps