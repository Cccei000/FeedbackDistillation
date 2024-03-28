# encoding=utf-8
import gc
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from chatgpt.experience_maker import (Experience, LocalInferExperienceMaker,
                                      StepExperience)
from chatgpt.nn.utils import (compute_approx_kl, compute_reward,
                              zero_pad_sequences)
from chatgpt.utils import is_rank_0, logging_rank_0


class StepLevelExperienceMaker(LocalInferExperienceMaker):
    """
    Experience maker for step-level PPO.
    """

    def __init__(self,
        **kwargs) -> None:
        """Step粒度的经验池采样类

        Args:
            **kwargs: 见'chatgpt.experience_maker.LocalInferExperienceMaker.__init__'
        """        
        
        super().__init__(**kwargs)

        #### 检查必要参数 ####
        if self.gs_args is None:
            logging_rank_0(f"Warning: Attribute 'enable_step_ppo' is not used in LocalInferExperienceMaker.", "debug")
            
        return

    def make_experience_with_actor_critic_and_rm(self, input_ids: torch.Tensor, is_val_make:bool=False) -> Dict[str, torch.Tensor]:
        """Actor、Critic推理：
            先使用Actor生成，再使用Actor和Critic推理得到logprobs和values。
            Actor生成时采用树搜索，同时进行了RM打分，获得了生成文本的奖励值。

        Args:
            input_ids (torch.Tensor): 输入Query的token ids

        Returns:
            Dict[str, torch.Tensor]: 推理结果
        """        
        
        # 验证时，不使用 RM 进行搜索
        if not is_val_make:
            self.reward_model.cuda()

        # 用于存储推理结果
        mini_batch_dict = defaultdict(list)
        mini_batch_response = {}
        
        with torch.no_grad():
            gen_batch_size = self.gen_minibatch_size if self.gen_minibatch_size else self.actor_minibatch_size
            for i in tqdm(range(0, len(input_ids), gen_batch_size),
                          desc=f"Generating",
                          disable=not is_rank_0()):
                mini_batch_input = input_ids[i: i + gen_batch_size]
                
                if not is_val_make:
                    # Actor生成mini-batch的样本
                    # 细粒度分步时返回分步reward delta
                    sequences, attention_mask, action_mask, fg_reward, step_reward, step_id = self.actor.module.generate(
                        input_ids=mini_batch_input,
                        gs_args=self.gs_args,
                        reward_model=self.reward_model,
                        for_validation=False,
                        **self.gen_args,
                    ) 
                    
                    # NOTE: 启动 do_finegrained 时，在生成时同步进行了打分，获得了奖励值
                    mini_batch_dict["fg_reward"].append(fg_reward)      # (bs_flatten,)
                    mini_batch_dict["step_reward"].append(step_reward)  # (bs_flatten, steps_num)
                    mini_batch_dict["step_id"].append(step_id)          # (bs_flatten,)
                else:
                    # Actor生成mini-batch的样本
                    # 细粒度分步时返回分步reward delta
                    sequences, attention_mask, action_mask = self.actor.module.generate(
                        input_ids=mini_batch_input,
                        gs_args=None,
                        reward_model=None,
                        for_validation=True,
                        **self.gen_args,
                    ) 
                    
                # 拆分sequence为steps
                # 分步更新时，generate返回多条样本
                mini_batch_dict["sequence"].append(sequences)               # (bs_flatten, seq_len)
                mini_batch_dict["attention_mask"].append(attention_mask)    # (bs_flatten, seq_len)
                mini_batch_dict["action_mask"].append(action_mask)          # (bs_flatten, seq_len)
                gc.collect()
                
            if not is_val_make:
                self.reward_model.cpu()
                torch.cuda.empty_cache()
                gc.collect()

            # 对齐生成的文本（统一右填充）
            # step_reward （左填充：GAE逆序计算）
            for key, value in mini_batch_dict.items():
                if key == "fg_reward":
                    mini_batch_response[key] = torch.cat(value, dim=0).to(self.actor.model.device)
                    continue
                if key == "step_id":
                    mini_batch_response[key] = torch.cat(value, dim=0).to(self.actor.model.device)
                    continue
                max_len = max(item.shape[1] for item in value)
                if key == "sequence":
                    padded_value = [
                        F.pad(item, (0, max_len - item.shape[1]), value=self.pad_token_id) # 使用传入的pad_token_id填充
                        for item in value
                    ]
                elif key == "step_reward":
                    padded_value = [
                        F.pad(item, (max_len - item.shape[1], 0), value=0)  # TODO：GAE计算使用0填充？？
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
                sequences = mini_batch_response["sequence"][start : end]                # (bs_flatten, seq_len)
                attention_mask = mini_batch_response["attention_mask"][start : end]
                num_actions = mini_batch_response["action_mask"].shape[1]
                action_mask = mini_batch_response["action_mask"][start : end]
                action_log_probs = self.actor(sequences, num_actions, attention_mask)   # (bs_flatten, seq_len - 1)
                value = self.critic(sequences, action_mask, attention_mask)             # (bs_flatten,)
                
                mini_batch_dict["action_log_probs"].append(action_log_probs)
                mini_batch_dict["value"].append(value)
        
            mini_batch_response["action_log_probs"] = torch.cat(mini_batch_dict["action_log_probs"], dim=0) #(num, seq_len - 1)
            mini_batch_response["value"] = torch.cat(mini_batch_dict["value"], dim=0)  #(num,)
            
        if not self.enable_rm_lora:
            self.reward_model.cpu()

        return mini_batch_response

    def process_step_level_reward(self, mini_batch_response:Dict[str,torch.Tensor], is_val_make: bool) -> List[Experience]:
        """整理分布奖励，进行Reward scaling，实例化经验池样本

        Args:
            mini_batch_response (Dict[str,torch.Tensor]): _description_
            is_val_make (bool): _description_

        Returns:
            List[Experience]: _description_
        """        
        
        sequences = mini_batch_response["sequence"]                             # (bs_flatten, seq_len)
        attention_mask = mini_batch_response["attention_mask"]                  # (bs_flatten, seq_len)
        action_mask = mini_batch_response["action_mask"]                        # (bs_flatten, seq_len)
        action_log_probs = mini_batch_response["action_log_probs"]              # (bs_flatten, seq_len - 1)
        base_action_log_probs = mini_batch_response["base_action_log_probs"]    # (bs_flatten, seq_len - 1)
        value = mini_batch_response["value"]                                    # (bs_flatten,)
        
        # 在Reward scaling之前，备份每个样本的奖励值
        origin_reward_all = mini_batch_response["fg_reward"].clone()            # (bs_flatten,)
        
        ## 对每个step的reward进行sacling
        ## NOTE: 注意！这里直接操作了mini_batch_response["fg_reward"]，所以在上面预先备份了一份没有scale的fg_reward
        if self.enable_reward_scaling:
            ## 获取每个样本action的起止index
            self.reward_scaler.reset()
            ## 遍历展开后的所有样本reward，scaling
            for item_idx in range(mini_batch_response["fg_reward"].shape[0]):
                mini_batch_response["fg_reward"][item_idx] = self.reward_scaler(mini_batch_response["fg_reward"][item_idx])
                ## 同步不同DP rank的scaler
            self.reward_scaler.sync()

        # 完成reward scaling的奖励值
        r = mini_batch_response["fg_reward"]                # (bs_flatten)
        # QUESTION: 这里的step_reward在后面被替换了，没有被用过
        step_reward = mini_batch_response["step_reward"]    # (bs_flatten, step_num)
        step_id = mini_batch_response["step_id"]            # (bs_flatten,)
        
        # 使用approx_kl修正奖励值
        reward = compute_reward(r, self.kl_coef, action_log_probs, base_action_log_probs, action_mask=action_mask) # (bs_flatten,)

        # 使用修正后的奖励值，重新计算step_reward，顺便计算step_value
        step_reward_kl = []
        reward_i = []
        value_i = []
        step_value = []
        for step_index in range(len(reward)):
            if step_index == 0 or step_id[step_index-1] == 1 or step_index == len(reward)-1:
                if step_index == len(reward)-1:
                    reward_i.append(reward[step_index])
                    value_i.append(value[step_index])
                for reward_i_index in range(len(reward_i)):
                    step_reward_kl.append(torch.tensor(reward_i))
                    step_value.append(torch.tensor(value_i))

                reward_i = []
                value_i = []
            reward_i.append(reward[step_index])
            value_i.append(value[step_index])

        step_reward = zero_pad_sequences(step_reward_kl, side="left", padding_value=0)
        step_reward = step_reward.to(self.actor.model.device)
        step_value = zero_pad_sequences(step_value, side="left", padding_value=0)
        step_value = step_value.to(self.actor.model.device)
        
        if is_val_make:
            value = mini_batch_response["value"]
            advantage = reward - value # (bs_flatten,)
        else:
            # 由后续步骤计算GAE
            advantage = torch.zeros_like(reward)    # (bs_flatten,)
            
        exps = [
            StepExperience(sequences=sequences,                     # (bs_flatten, seq_len)
                            action_log_probs=action_log_probs,       # (bs_flatten, seq_len-1)
                            values=value,                            # (bs_flatten,)
                            reward=reward,                           # (bs_flatten,)
                            advantages=advantage,                    # (bs_flatten,)
                            attention_mask=attention_mask,           # (bs_flatten, seq_len)
                            action_mask=action_mask,                 # (bs_flatten, seq_len-1)
                            origin_reward=origin_reward_all,         # (bs_flatten,)
                            step_reward=step_reward,                 # (bs_flatten, step_num)
                            step_value=step_value,                   # (bs_flatten, step_num)
                            step_id=step_id                          # (bs_flatten, step_num)    
            )
        ]

        return exps

    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor, return_kl:bool=False, is_val_make=False, step:int=0, **kwargs) -> List[Experience]:

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

        # 先做actor critic rm forward
        mini_batch_response = self.make_experience_with_actor_critic_and_rm(input_ids=input_ids, is_val_make=is_val_make)
        gc.collect()
        # 再做init_model forward
        batch_base_action_log_probs = self.make_experience_with_initial_model(mini_batch_response)
        mini_batch_response["base_action_log_probs"] = batch_base_action_log_probs
        gc.collect()
        # 处理奖励值，构造经验池
        if is_val_make:
            # 验证流程不需要分步，所以调用父类方法打分
            exps = super().make_experience_with_reward_model(mini_batch_response=mini_batch_response)
        else:
            exps = self.process_step_level_reward(mini_batch_response, is_val_make=is_val_make)
        gc.collect()
        
        if return_kl:
            approx_kl = compute_approx_kl(
                mini_batch_response["action_log_probs"], mini_batch_response["base_action_log_probs"], action_mask=mini_batch_response["action_mask"], return_mean=True
            )
            # print(f"KL:{approx_kl.mean().item()}")
            # if approx_kl.mean().item() > 1:
            #     print(f'ERROR: {mini_batch_response["sequence"].tolist()} | {mini_batch_response["action_log_probs"].tolist()} | {mini_batch_response["base_action_log_probs"].tolist()}')

            return exps, approx_kl.mean()
            
        return exps
    
    def logging(self, step):
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
            return self.reward_model.recover_reward(reward_mean, reward_std, is_token)
        except:
            logging_rank_0(f"Reward don't need to be recovered.", "debug")
        
        return reward_mean, reward_std
    