# encoding=utf-8
import gc
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from chatgpt.experience_maker import (Experience, ExperienceMaker,
                                      StepExperience)
from chatgpt.nn import Actor, Critic, GSArgs, RewardModel
from chatgpt.nn.utils import (compute_approx_kl, compute_reward,
                              zero_pad_sequences)
from chatgpt.pipeline.config import ActorGranularity, RewardModelGranularity
from chatgpt.utils import (ACTOR_INFER, CRITIC_INFER, GENERATE, INFER_MODELS,
                           REF, RM, CostTimer, DeepspeedFlopsTimerGroup,
                           FlopsTimerGroup, LoggingLevel, is_rank_0,
                           logging_rank_0)

from .reward_scaling import RewardScaling


class LocalInferExperienceMaker(ExperienceMaker):
    """
    Naive experience maker.
    """

    def __init__(self,
        actor: Actor,
        critic: Critic,
        reward_model: RewardModel,
        initial_model: Actor,
        gen_args:dict,
        rm_granularity: RewardModelGranularity,
        ppo_granularity: ActorGranularity,
        kl_coef: float = 0.1,
        seed: int = 1234,
        pad_token_id: int = 2,
        eos_token_id: int = 2,
        max_seq_len:int=2048,
        actor_minibatch_size: int = 16,
        rm_minibatch_size: int = 1,
        gen_minibatch_size: int = 1,
        gs_args:Optional[GSArgs]=None,
        enable_reward_scaling:bool=False,
        reward_scaling_gamma:float=0.95,
        use_delta_reward:bool=False,
        enable_rm_lora:bool=False,
        enable_policy_lora:bool=False,
        logger=None,
        flops_timers:Optional[FlopsTimerGroup]=None,
        deepspeed_flops_timers:Optional[DeepspeedFlopsTimerGroup]=None,
        **kwargs,) -> None:
        """基础的经验池采样类

        Args:
            actor (Actor): Actor模型
            critic (Critic): Critic模型
            reward_model (RewardModel): 奖励模型
            initial_model (Actor): 固定的初始Actor模型
            gen_args (dict): Generate的配置.
            rm_granularity (RewardModelGranularity): RM训练阶段采用的粒度
            ppo_granularity (ActorGranularity): PPO训练采用的粒度
            kl_coef (float, optional): Approx kl 惩罚系数. Defaults to 0.1.
            seed (int, optional): 随机种子（需要保证MP group、PP group内的种子相同）. Defaults to 1234.
            pad_token_id (int, optional): Actor采用的pad token id. Defaults to 2.
            eos_token_id (int, optional): Actor采用的eos token id. Defaults to 2.
            max_seq_len (int, optional): 经验池样本的最大token长度，用于右截断过长样本. Defaults to 2048.
            actor_minibatch_size (int, optional): Actor模型forward采用的mini-batch大小. Defaults to 16.
            rm_minibatch_size (int, optional): 奖励模型推理采用的mini-batch大小. Defaults to 1.
            gen_minibatch_size (int, optional): Actor模型生成文本采用的mini-batch大小. Defaults to 1.
            gs_args (Optional[GSArgs], optional): Generate with Search的配置，详见'chatgpt.nn.generate_search.py'. Defaults to None.
            enable_reward_scaling (bool, optional): 是否启用Reward Scaling. Defaults to False.
            reward_scaling_gamma (float, optional): Reward Scaling采用的折扣系数. Defaults to 0.95.
            use_delta_reward (bool, optional): 是否计算相邻token奖励的差值作为实际的奖励（仅在enable_gae并且token_level_reward有效）. Defaults to False.
            enable_rm_lora (bool, optional): 是否使用基于LoRA训练的奖励模型（影响模型的上下显存）. Defaults to False.
            enable_policy_lora (bool, optional): 是否基于LoRA训练Actor、Critic（影响模型的上下显存）. Defaults to False.
            logger (_type_, optional): Logger. Defaults to None.
            flops_timers (Optional[FlopsTimerGroup], optional): flops_timers. Defaults to None.
            deepspeed_flops_timers (Optional[DeepspeedFlopsTimerGroup], optional):  Deepspeed提供的flops_timers. Defaults to None.

        Raises:
            AttributeError: _description_
            AttributeError: _description_
        """      
        
        
        
        
        super().__init__(actor, critic, reward_model, initial_model, kl_coef)
        
        torch.manual_seed(seed)
        
        ### Basic param ###
        self.seed                   = seed
        self.pad_token_id           = pad_token_id
        self.eos_token_id           = eos_token_id
        self.max_seq_len            = max_seq_len
        
        ### mini-batch size ### 
        self.actor_minibatch_size   = actor_minibatch_size
        self.gen_minibatch_size     = gen_minibatch_size
        self.rm_minibatch_size      = rm_minibatch_size
        
        ### granularity ###
        self.ppo_granularity        = ppo_granularity
        self.rm_granularity         = rm_granularity
        
        ### generation param ###
        self.gs_args                = gs_args
        self.gen_args               = gen_args
        
        ### loggers and profilers ###
        self.logger                 = logger
        # self.reward_model.logger    = logger
        self.flops_timers           = flops_timers
        self.deepspeed_flops_timers = deepspeed_flops_timers
        self.enabling_flops_recording = flops_timers is not None
        
        ### Algorithm param ###
        self.use_delta_reward       = use_delta_reward
        self.enable_reward_scaling  = enable_reward_scaling
        self.reward_scaler = RewardScaling(shape=1, device=actor.device, gamma=reward_scaling_gamma) if enable_reward_scaling else None
        
        logging_rank_0(f"{'Enable' if enable_reward_scaling else 'Disable'} Reward Scaling", "debug")
        
        ### LoRA param ###
        self.enable_rm_lora         = enable_rm_lora
        self.enable_policy_lora     = enable_policy_lora
        if self.enable_rm_lora:
            logging_rank_0(f"Not applicable to 'LoRA RM'. Set to False.", "warning")
            self.enable_rm_lora = False
        
        ### unused param ###
        for key in kwargs:
            if key == "is_lora": 
                logging_rank_0(
                    f"Deprecation Warning: Attribute 'is_lora' is not used in LocalInferExperienceMaker. Please use 'enable_rm_lora' and 'enable_policy_lora'!",
                    LoggingLevel.DEBUG
                )
            else:
                logging_rank_0(f"Deprecation Warning: Attribute '{key}' is not used in LocalInferExperienceMaker.", LoggingLevel.DEBUG)
        
        ### ExperienceMaker Info ###
        self.enable_searching = False
        if gs_args is None:
            logging_rank_0(f"ExperienceMaker do vanilla search", level='debug')
        elif gs_args.enabling_tot:
            if ppo_granularity is ActorGranularity.token:
                logging_rank_0(f"Generating with tot searching is not available for token-level PPO.", "error")
                raise AttributeError
            logging_rank_0(f"ExperienceMaker do bfs generating", level='debug')
            self.enable_searching = True
        elif gs_args.enabling_bon:
            if ppo_granularity is ActorGranularity.token:
                logging_rank_0(f"Generating with best-of-n searching is not available for token-level PPO.", "error")
                raise AttributeError
            logging_rank_0(f"ExperienceMaker do best-of-n search", level='debug')
            self.enable_searching = True
        else:
            logging_rank_0(f"ExperienceMaker do vanilla search", level='debug')
        return

    def make_experience_with_actor_critic(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Actor、Critic推理：
            先使用Actor生成，再使用Actor和Critic推理得到logprobs和values

        Args:
            input_ids (torch.Tensor): 输入Query的token ids

        Returns:
            Dict[str, torch.Tensor]: 推理结果
        """        

        # 用于存储推理结果
        mini_batch_dict = defaultdict(list)
        mini_batch_response = {}

        # 如果启用tot_search或者best_of_N生成，需要预先将RM移至显存
        if self.enable_searching:
            self.reward_model = self.reward_model.cuda()
            self.enabling_flops_recording = False   # flops timer 暂不支持树搜索生成

        with torch.no_grad():
            gen_batch_size = self.gen_minibatch_size if self.gen_minibatch_size else self.actor_minibatch_size
            for i in tqdm(range(0, len(input_ids), gen_batch_size),
                          desc=f"Generating",
                          disable=not is_rank_0()):
                mini_batch_input = input_ids[i: i + gen_batch_size]
                
                self._on_forward_start(model_name=GENERATE)
                sequences, attention_mask, action_mask, gen_info = self.actor.module.generate(
                    input_ids=mini_batch_input,
                    gs_args=self.gs_args,
                    reward_model=self.reward_model,
                    **self.gen_args
                ) 
                self._on_forward_end(model_name=GENERATE, time=gen_info["total_time"], batch_info=gen_info["total_tokens"])

                gc.collect()
                mini_batch_dict["sequence"].append(sequences)               # (bs, seq_len)
                mini_batch_dict["attention_mask"].append(attention_mask)    # (bs, seq_len)
                mini_batch_dict["action_mask"].append(action_mask)          # (bs, seq_len)
                
            if self.enable_searching:
                self.reward_model = self.reward_model.cpu()

            # 对齐生成的文本（统一右填充）
            for key, value in mini_batch_dict.items():
                if key == "fg_reward" or key == "step_id" or key == "step_reward":
                    continue
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

            # actor 和 critic 推理
            for i in tqdm(range(0, mini_batch_response["sequence"].shape[0], self.actor_minibatch_size),
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
                value = self.critic(sequences, action_mask, attention_mask)             # (bs, seq_len - 1) or (bs,)
                self._on_forward_end(CRITIC_INFER, batch_size=sequences.shape[0], seq_length=sequences.shape[1], time=CostTimer.get_time())

                mini_batch_dict["action_log_probs"].append(action_log_probs)
                mini_batch_dict["value"].append(value)
        
            mini_batch_response["action_log_probs"] = torch.cat(mini_batch_dict["action_log_probs"], dim=0) #(num, seq_len - 1)
            mini_batch_response["value"] = torch.cat(mini_batch_dict["value"], dim=0)  #(num,)

        return mini_batch_response
    
    
    def make_experience_with_initial_model(self, mini_batch_response:Dict[str,torch.Tensor]) -> torch.Tensor:
        """SFT模型推理，获得初始模型在生成样本上的logprobs

        Args:
            mini_batch_response (Dict[str,torch.Tensor]): _description_

        Returns:
            torch.Tensor: _description_
        """         

        self.initial_model.cuda()
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
                
                self._on_forward_start(REF)
                base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)  # (bs, seq_len - 1)
                self._on_forward_end(REF, batch_size=sequences.shape[0], seq_length=sequences.shape[1], time=CostTimer.get_time())
                
                batch_base_action_log_probs.append(base_action_log_probs)
        if not self.enable_policy_lora:
            self.initial_model.cpu()

        # 因为输入已经padding过了，此时base_action_log_probs是已经对齐了的
        batch_base_action_log_probs = torch.cat(batch_base_action_log_probs, dim=0)

        return batch_base_action_log_probs

    def make_experience_with_reward_model(self, mini_batch_response:Dict[str,torch.Tensor]) -> List[Experience]:
        """奖励模型推理

        Args:
            mini_batch_response (Dict[str,torch.Tensor]): _description_

        Returns:
            List[Experience]: _description_
        """        
        
        self.reward_model.cuda()
        total_num = mini_batch_response["sequence"].shape[0]
        exps = []
                
        for i in tqdm(range(0, total_num, self.rm_minibatch_size), 
                        desc=f"Scoring",
                        disable=not is_rank_0()):

            # 划分 mini-batch
            start, end = i, i + self.rm_minibatch_size
            sequences = mini_batch_response["sequence"][start : end][:self.max_seq_len]                             # (bs, seq_len)
            attention_mask = mini_batch_response["attention_mask"][start : end][:self.max_seq_len]                  # (bs, seq_len)
            action_mask = mini_batch_response["action_mask"][start : end][:self.max_seq_len-1]                      # (bs, seq_len - 1)
            action_log_probs = mini_batch_response["action_log_probs"][start : end][:self.max_seq_len-1]            # (bs, seq_len - 1)
            base_action_log_probs = mini_batch_response["base_action_log_probs"][start : end][:self.max_seq_len-1]  # (bs, seq_len - 1)
            if mini_batch_response["value"].dim == 1:
                value = mini_batch_response["value"][start : end]                           # (bs,)
            else:
                value = mini_batch_response["value"][start : end][:self.max_seq_len-1]      # (bs, seq_len - 1)
                
            with torch.no_grad():
                # RM Forward
                self._on_forward_start(RM)
                r = self.reward_model(sequences, action_mask, attention_mask) # (bs,) or (bs, seq_len)
                self._on_forward_end(RM, batch_size=sequences.shape[0], seq_length=sequences.shape[1], time=CostTimer.get_time())

            # Sample-level PPO
            if self.ppo_granularity is ActorGranularity.sample:
                origin_reward = r.detach().clone()
                reward = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask)
                advantage = reward - value # (bs,)
            # Token-level PPO
            elif self.ppo_granularity is ActorGranularity.token:
                # Sample-level RM
                if self.rm_granularity is RewardModelGranularity.sample:
                    # 初始化样本 token 奖励值
                    token_reward = torch.zeros_like(action_log_probs, dtype=r.dtype)    # (bs, seq_len - 1)
                    # 填入 last token 奖励
                    last_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in attention_mask[:, 1:]], device=token_reward.device, dtype=torch.long)
                    item_index = torch.arange(0, token_reward.shape[0], device=token_reward.device, dtype=torch.long)
                    token_reward[item_index, last_index] = r   
                    # 计算 sft model approx kl 
                    approx_kl = compute_approx_kl(log_probs=action_log_probs, log_probs_base=base_action_log_probs, return_mean=False, action_mask=action_mask)  # (bs, seq_len - 1)
                    # 计算 approx kl 修正的奖励
                    origin_reward = token_reward.detach().clone()       # (bs, seq_len - 1)
                    reward = token_reward + self.kl_coef * approx_kl    # (bs, seq_len - 1)
                    # 初始化 advantage，置零，供后续 GAE
                    advantage = torch.zeros_like(reward)                # (bs, seq_len - 1)
                    
                # Token-level RM or Token-mix-sample RM
                else:
                    # 计算 sft model approx kl 
                    approx_kl = compute_approx_kl(log_probs=action_log_probs, log_probs_base=base_action_log_probs, return_mean=False, action_mask=action_mask)  # (bs, seq_len - 1)
                    # 计算 token-level delta 奖励
                    if self.use_delta_reward:
                        reward_delta = r[:, 1:] - r[:, :-1]
                    else:
                        reward_delta = r[:, 1:]
                    # 计算 approx kl 修正的奖励
                    origin_reward = reward_delta.detach().clone()
                    reward = reward_delta + self.kl_coef * approx_kl 
                    
                    ## 对每个token的reward进行sacling
                    if self.enable_reward_scaling:
                        ## 获取每个样本action的起止index
                        last_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in attention_mask[:, 1:]], device=reward.device, dtype=torch.long)
                        start_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][0].item() for a in action_mask], device=reward.device, dtype=torch.long)
                        ## 遍历每个样本，从start token开始scaling
                        for item_idx in range(reward.shape[0]):
                            idx_st = start_index[item_idx]
                            idx_nd = last_index[item_idx]
                            self.reward_scaler.reset()
                            for token_idx in range(idx_st, idx_nd + 1):
                                reward[item_idx][token_idx] = self.reward_scaler(reward[item_idx][token_idx])
                            ## 同步不同DP rank的scaler
                            self.reward_scaler.sync()
                    
                    advantage = torch.zeros_like(reward)
            # Step-level PPO（支持StepLevelExperienceMaker的验证流程）
            else:
                origin_reward = r.detach().clone()
                reward = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask)
                advantage = reward - value # (bs,)

            
            if advantage.ndim == 1:
                advantage = advantage.unsqueeze(-1)

            exps.append(Experience(sequences, action_log_probs, value, reward, advantage, attention_mask, action_mask, origin_reward=origin_reward))

        if not self.enable_rm_lora:
            self.reward_model.cpu()

        return exps

    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor, return_kl:bool=False, step:int=0, **kwargs) -> List[Experience]:
        """采样经验池

        Args:
            input_ids (torch.Tensor): 输入Query的token id，若pure_input_ids不为None，则其中包含guidance的回答
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
        mini_batch_response = self.make_experience_with_actor_critic(input_ids=input_ids)
        gc.collect()
        # 再做init_model forward
        batch_base_action_log_probs = self.make_experience_with_initial_model(mini_batch_response)
        mini_batch_response["base_action_log_probs"] = batch_base_action_log_probs
        gc.collect()
        # 再做rw forward
        exps = self.make_experience_with_reward_model(mini_batch_response)
        gc.collect()
        
        #############
        self._on_profile_flops(step=step, model_name=INFER_MODELS)
        #############
        # 
        if return_kl:
            approx_kl = compute_approx_kl(
                mini_batch_response["action_log_probs"], mini_batch_response["base_action_log_probs"], action_mask=mini_batch_response["action_mask"], return_mean=True
            )
            
            return exps, approx_kl.mean()
            
        return exps
    
    def logging(self, step:int):
        if self.enable_reward_scaling:
            logging_rank_0(f"log reward scaler.", "debug")
            self.logger.log_metrics(
                {
                    "mean": self.reward_scaler.mean.item(),
                    "std": self.reward_scaler.std.item()
                }, step=step, metrics_group='reward_scaling')
        
        try:
            self.reward_model.logging(step)
        except:
            logging_rank_0(f"Reward Model needn't logging.", "debug")
        
        return
    
    def recover_reward(self, reward_mean:torch.Tensor, reward_std:torch.Tensor, is_token:bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            reward_mean, reward_std = self.reward_model.recover_reward(reward_mean, reward_std, is_token)
        except:
            reward_mean, reward_std = reward_mean, reward_std
        
        return reward_mean, reward_std
    
    
    def _on_forward_start(self, model_name:str) -> None:
        if not self.enabling_flops_recording:
            return
        # 记录生成的计算量
        if self.deepspeed_flops_timers is not None:
            self.deepspeed_flops_timers.end_profile(model_name)     # NOTE:是否每次记录都清零？
            self.deepspeed_flops_timers.start_profile(model_name)
            
        return

    def _on_forward_end(self, model_name:str, **kwargs) -> None:
        
        if not self.enabling_flops_recording:
            return
             
        seq_length = kwargs.get("seq_length", None)
        batch_size = kwargs.get("batch_size", None)
        batch_info = kwargs.get("batch_info", None)
        iter_time_s = kwargs.get("time", None)
        
        # 记录生成的计算量
        if self.deepspeed_flops_timers is not None:
            self.deepspeed_flops_timers.stop_profile(model_name)
        if self.flops_timers is not None:
            if iter_time_s is not None and batch_info is not None:
                self.flops_timers.reset(model_name)
                self.flops_timers.update_timer(model_name=model_name, iter_time_s=iter_time_s)
                self.flops_timers.update_calculation(model_name=model_name, batch_info=batch_info)
            elif iter_time_s is not None and seq_length is not None and batch_size is not None:
                self.flops_timers.reset(model_name)
                self.flops_timers.update_timer(model_name=model_name, iter_time_s=iter_time_s)
                self.flops_timers.update_calculation(model_name=model_name, batch_size=batch_size, seq_length=seq_length)
        
        return
            
    def _on_profile_flops(self, step:int, model_name:Optional[str]=INFER_MODELS) -> None:
        
        if not self.enabling_flops_recording:
            return
        
        if self.flops_timers is not None:
            flops_dict = self.flops_timers.get_flops(model_name)
            log_info = {
                f"{name}": flops
                for name, flops in flops_dict.items()
            }
            log_info["infer_all"] = self.flops_timers.get_avg_flops(model_name=model_name)
            self.logger.log_metrics(log_info, step=step, metrics_group='flops')
            
        if self.deepspeed_flops_timers is not None:
            flops_dict = self.deepspeed_flops_timers.get_flops(model_name)
            log_info = {
                f"{name}": flops
                for name, flops in flops_dict.items()
            }
            log_info["infer_all"] = self.deepspeed_flops_timers.get_avg_flops(model_name=model_name)
            self.logger.log_metrics(log_info, step=step, metrics_group='flops_deepspeed')

        return
        
