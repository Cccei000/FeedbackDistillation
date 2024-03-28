from typing import Callable, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from chatgpt.experience_maker import Experience, PPOPPExperienceMaker
from chatgpt.nn.utils import (get_global_statistics, masked_mean,
                              zero_pad_sequences)
from chatgpt.pipeline.config import ActorGranularity, RewardModelGranularity
from chatgpt.trainer import PPOTrainer
from chatgpt.utils import is_rank_0, logging_rank_0


class PPOPPTrainer(PPOTrainer):
    """
        Trainer for PPO algorithm.

    Args:
        actor (Actor): the actor model in ppo algorithm
        critic (Critic): the critic model in ppo algorithm
        reward_model (nn.Module): the reward model in rlhf algorithm to make reward of sentences
        initial_model (Actor): the initial model in rlhf algorithm to generate reference logits to limit the update of actor
        actor_optim (Optimizer): the optimizer to use for actor model
        critic_optim (Optimizer): the optimizer to use for critic model
        train_batch_size (int, defaults to 8): the batch size to use for training
        buffer_limit (int, defaults to 0): the max_size limitaiton of replay buffer
        buffer_cpu_offload (bool, defaults to True): whether to offload replay buffer to cpu
        eps_clip (float, defaults to 0.2): the clip coefficient of policy loss
        value_clip (float, defaults to 0.4): the clip coefficient of value loss
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        tokenier (Callable, optional): the tokenizer to use for tokenizing the input
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    """

    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)
        
        ###
        if self.ppo_granularity is ActorGranularity.step:
            logging_rank_0(f"'PPOPPTrainer' is not available for step-level PPO. Please use 'PPOTrainer'.", "error")
            raise NotImplementedError
        if not isinstance(self.experience_maker, PPOPPExperienceMaker):
            logging_rank_0(f"'{type(self.experience_maker)}' is not available for PPOPP. Please use 'PPOPPExperienceMaker'.", "error")
            raise NotImplementedError
        return
    
    def fit(self,
            prompts:Dict[str,List[str]],
            guided_prompts:Dict[str,List[str]],
            val_prompts:Dict[str,List[str]],
            guidance:Dict[str,str],
            guidance_beta:float=0.2,
            guidance_beta_decay:float=0.98,
            guidance_init_rate:float=0.8,
            guidance_rate_decay:float=0.98,
            seed: int = 42,
            num_episodes: int = 50000,
            max_timesteps: int = 500,
            update_timesteps: int = 5000,
            val_check_interval:int=2,
            val_saving_func:Optional[Callable] = None) -> None:
        """_summary_

        Args:
            prompts (Dict[str,List[str]]): 用于训练的所有prompt，根据任务类型组织
            guided_prompts (Dict[str,List[str]]): 用于训练的所有有引导的prompt，根据任务类型组织
            val_prompts (Dict[str,List[str]]): 用于验证的所有prompt，根据任务类型组织
            guidance (Dict[str,str]): prompt的引导response，键名为prompt，键值为引导response
            guidance_beta (float, optional): 引导样本比例. Defaults to 0.2.
            guidance_init_rate (float, optional): 单条引导样本中，使用的引导response文本比例（初始值）.Defaults to 0.8.
            guidance_rate_decay (float, optional): guidance_rate衰减比率. Defaults to 0.98.
            seed (int, optional): 锁定的随机种子. Defaults to 42.
            num_episodes (int, optional): 总训练轮次. Defaults to 50000.
            max_timesteps (int, optional): _description_. Defaults to 500.
            update_timesteps (int, optional): _description_. Defaults to 5000.
            val_check_interval (int, optional): _description_. Defaults to 2.
            val_saving_func (Optional[Callable], optional): _description_. Defaults to None.
        """
        
        # 启动验证集的pipeline
        time = 0
        self._on_fit_start()
        
        # 预处理验证集
        val_prompt_text = []
        val_task_names = []
        for task, prompt in val_prompts.items():
            val_prompt_text.extend(prompt)
            val_task_names.extend([task]*len(prompt))
        
        # episode循环
        for episode in range(num_episodes):
            # 验证
            if episode % val_check_interval == 0:
                
                logging_rank_0("Validation")
                inputs = self.tokenizer.batch_encode_plus(val_prompt_text)["input_ids"]
                self._on_make_experience_start()
                self.make_exp_steps += 1
                experiences = self._make_experience(inputs, is_val=True)
                self._log_val_metrics(experiences=experiences, val_prompts=val_prompts)
                self.replay_buffer.append(experiences)
                
                logging_rank_0(f"Saving Checkpoint on episode {episode}")
                self.ckpt_saving_func(episode, self.actor)
                if val_saving_func is not None:
                    val_saving_func(self.replay_buffer, episode, val_task_names)
                self.replay_buffer.clear()
                self._on_episode_end(episode=episode)
                continue
            
            trained_episode = int(episode / val_check_interval) * (val_check_interval - 1) + (episode % val_check_interval) - 1
            curr_guidance_rate = guidance_init_rate * (guidance_rate_decay**trained_episode)
            curr_guidance_beta = guidance_beta * (guidance_beta_decay**trained_episode)
            
            logging_rank_0(f"Rate:{curr_guidance_rate} | Beta: {curr_guidance_beta}", "debug")
            
            self.timers('episode').start()
            self._on_episode_start(episode)
            # 采样训练
            for timestep in tqdm(range(max_timesteps),
                                desc=f'Episode [{episode+1}/{num_episodes}]',
                                disable=not is_rank_0()):
                time += 1
                free_prompts, guide_prompts = self._sample_prompts(
                    prompts=prompts,
                    guided_prompts=guided_prompts,
                    guidance=guidance,
                    guidance_beta=curr_guidance_beta,
                    guidance_rate=curr_guidance_rate,
                    seed=seed+timestep+episode
                )

                # Encode Free Prompts
                free_inputs = self.tokenizer.batch_encode_plus(free_prompts)["input_ids"]
                # Encode Guide Prompts
                guide_inputs = []
                pure_inputs = []
                for curr, prev in guide_prompts:
                    curr_ids = self.tokenizer(curr)["input_ids"]
                    prev_ids = self.tokenizer(prev)["input_ids"]
                    guide_len = max(1, int((len(curr_ids) - len(prev_ids)) * curr_guidance_rate)) + len(prev_ids)
                    guide_inputs.append(curr_ids[:guide_len])
                    pure_inputs.append(prev_ids)
                
                self.timers('make_experience').start()
                
                self.make_exp_steps += 1
                
                assert len(guide_inputs) > 0 or len(free_inputs) > 0
                
                # Sample Guide
                if len(guide_inputs) > 0:
                    self._on_make_experience_start()
                    experiences = self._make_experience(guide_inputs, pure_inputs, is_guide=True)
                    self._on_make_experience_end(experiences, is_guide=True)
                    self.replay_buffer.append(experiences)
                
                # Sample Free
                if len(free_inputs) > 0:
                    self._on_make_experience_start()
                    experiences = self._make_experience(free_inputs, is_guide=False)
                    self._on_make_experience_end(experiences, is_guide=False)
                    self.replay_buffer.append(experiences)
                
                self.timers('make_experience').stop()
                
                # 启动训练
                if time % update_timesteps == 0:
                    make_exp_time = self.timers('make_experience').elapsed(reset=True)
                    self.logger.log_metrics({'make_exp_samples_per_second': len(self.replay_buffer)/make_exp_time},
                                            step=self.global_steps, metrics_group='timers')

                    self._on_learn_start()
                    self._learn(episode, timestep)
                    self._on_profile_flops(step=self.global_steps)
                        
                    self.replay_buffer.clear()                   

            self._on_episode_end(episode)
            self.timers('episode').stop()
            self.timers.write(['episode'], self.global_steps, reset=True, metrics_group='timers')
            
        self._on_fit_end()

    def _sample_prompts(self, prompts:Dict[str,List[str]], guided_prompts:Dict[str,List[str]], guidance:Dict[str,str], seed, guidance_beta:float, guidance_rate:float) -> Tuple[List[str],List[Tuple[str,str]]]:
        """
            随机选取用于生成的Prompt
            考虑提供的引导回答，拼接至原始prompt后

        Args:
            prompts (Dict[str,List[str]]): _description_
            guided_prompts (Dict[str,str]): _description_
            guidance (Dict[str,str]): _description_
            seed (_type_): _description_
            guidance_beta (float): _description_
            guidance_rate (float): _description_

        Returns:
            Tuple[List[str],List[Tuple[str,str]]]: _description_
        """
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        # 计算每个task需要采样的prompt数量
        ## guide样本总数
        num_guide = int(guidance_beta * self.experience_batch_size)
        ## 任务总数
        num_guide_task, num_free_task = len(guided_prompts), len(prompts)
        ## 每个guide数据集的guide样本数量
        num_guide_data_per_guide_task = int(num_guide / num_guide_task) if num_guide_task > 0 else 0
        ## 每个数据集的样本数量
        num_data_per_task = int(self.experience_batch_size / (num_guide_task + num_free_task))
        ## 每个guide数据集的guide样本数量，限制于每个数据集的样本数量
        num_guide_data_per_guide_task = min(num_data_per_task, num_guide_data_per_guide_task)
        ## 每个guide数据集的free样本数量
        num_free_data_per_guide_task = num_data_per_task - num_guide_data_per_guide_task
        ## 每个free数据集的样本数量
        num_data_per_free_task = int(
            (self.experience_batch_size - num_guide_task * (num_guide_data_per_guide_task + num_free_data_per_guide_task)) / num_free_task
        ) if num_free_task > 0 else 0
        
        output_free_prompts = []
        output_guide_prompts = []
        
        logging_rank_0(f"TOTAL: {self.experience_batch_size} | GG: {num_guide_data_per_guide_task} | FG: {num_free_data_per_guide_task} | FF: {num_data_per_free_task}", "debug")
        
        # Guide task
        for t_name, queries in guided_prompts.items():
            sampled_indices = torch.randperm(len(queries), generator=g).tolist()
            
            if num_free_data_per_guide_task > 0:
                free_indiced = sampled_indices[-num_free_data_per_guide_task:]
                selected_free_prompts = [queries[i] for i in free_indiced]
                output_free_prompts.extend(selected_free_prompts)
            
            guide_indices = sampled_indices[:num_guide_data_per_guide_task]
            selected_guide_prompts = [queries[i] for i in guide_indices]
            selected_guide_responses = [guidance[prompt] for prompt in selected_guide_prompts]
            concat_guide_prompts = [
                (f"{prev_prompt}{guide_response}", prev_prompt) 
                for prev_prompt, guide_response in zip(selected_guide_prompts, selected_guide_responses)
            ]   # 保留原始的prompt用于encode后截取
            output_guide_prompts.extend(concat_guide_prompts)
            
        # Free task
        for t_name, queries in prompts.items():
            sampled_indices = torch.randperm(len(queries), generator=g).tolist()[:num_data_per_free_task]
            selected_free_prompts = [queries[i] for i in sampled_indices]
            output_free_prompts.extend(selected_free_prompts)
    
        # logging
        self.logger.log_metrics(
            {
                "curr_beta": len(output_guide_prompts)/(len(output_free_prompts) + len(output_guide_prompts)),
                "curr_rate": guidance_rate
            },
            step=self.global_steps, metrics_group='guidance'
        )
        
        return output_free_prompts, output_guide_prompts
    
    def _make_experience(self, inputs: List[List[int]], pure_inputs: Optional[List[List[int]]]=None, is_val:bool=False, is_guide:bool=False) -> List[Experience]:
        """采样经验池

        Args:
            inputs (List[List[int]]): _desc一批query的input_idsription_
            pure_inputs (Optional[List[List[int]]], optional): _description_. Defaults to None.
            is_val (bool, optional): 是否是验证集. Defaults to False.
            is_guide (bool, optional): 是否是引导集. Defaults to False.

        Returns:
            List[Experience]: _description_
        """      
        
        input_ids = [torch.tensor(ids) for ids in inputs] # 转换为torch.Tensor
        
        # right padding
        inputs = zero_pad_sequences(input_ids, side="right", # side="left",
                                    padding_value=self.pad_token_id)
        
        if pure_inputs is not None:
            pure_input_ids = [torch.tensor(ids) for ids in pure_inputs] # 转换为torch.Tensor
            pure_inputs = zero_pad_sequences(pure_input_ids, side="right", # side="left",
                                        padding_value=self.pad_token_id)
            
        # 调用experience maker，采样经验池
        exp, kl_ref = self.experience_maker.make_experience(inputs, pure_inputs, return_kl=True, step=self.global_steps)
        
        # 计算本批经验池的Approx KL
        kl_ref_mean, _, _ = get_global_statistics(kl_ref)
        if is_val:
            self.logger.log_metrics({"kl_ref_val": kl_ref_mean.item()}, step=self.global_steps, metrics_group='experience')
        elif is_guide:
            self.logger.log_metrics({"kl_ref_guide": kl_ref_mean.item()}, step=self.global_steps, metrics_group='experience')
        else:
            self.logger.log_metrics({"kl_ref": kl_ref_mean.item()}, step=self.global_steps, metrics_group='experience')
        return exp

    def _on_make_experience_end(self, experiences: List[Experience], is_guide:bool=False) -> None:
        """采样完经验池后记录奖励值信息

        Args:
            experiences (List[Experience]): _description_

        Returns:
            _type_: _description_
        """
        
        assert len(experiences) > 0
        
        ## 计算本批experience的奖励值信息
        # Token-level PPO:
        if self.ppo_granularity is ActorGranularity.token:
            return_list = []
            reward_list = []
            sample_level_reward_list, sample_level_return_list = [], []
            # Sample-level RM
            if self.rm_granularity is RewardModelGranularity.sample:
                for exp in experiences:
                    last_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in exp.attention_mask[:, 1:]], device=exp.reward.device, dtype=torch.long)
                    reward_list.append(exp.origin_reward[torch.arange(0,exp.origin_reward.shape[0], device=exp.origin_reward.device), last_index])
                    return_list.append(exp.reward[torch.arange(0,exp.reward.shape[0], device=exp.reward.device), last_index])
            # Token-mix-sample RM
            elif self.rm_granularity is RewardModelGranularity.token_mix_sample:
                for exp in experiences:
                    last_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in exp.attention_mask[:, 1:]], device=exp.reward.device, dtype=torch.long)
                    sample_reward = exp.origin_reward[torch.arange(0,exp.origin_reward.shape[0], device=exp.origin_reward.device), last_index]
                    sample_return = exp.reward[torch.arange(0,exp.reward.shape[0], device=exp.reward.device), last_index]
                    token_reward = masked_mean(exp.origin_reward, exp.action_mask)
                    token_return = masked_mean(exp.reward, exp.action_mask)
                    # token_reward = token_reward - sample_reward ## 分离token sample
                    sample_level_reward_list.append(sample_reward)
                    sample_level_return_list.append(sample_return)
                    reward_list.append(token_reward)
                    return_list.append(token_return)
            # Token-level RM
            else:
                for exp in experiences:
                    reward_list.append(masked_mean(exp.origin_reward, exp.action_mask))
                    return_list.append(masked_mean(exp.reward, exp.action_mask))
        
        # Sample-level PPO
        else:
            reward_list = [exp.origin_reward for exp in experiences]
            return_list = [exp.reward for exp in experiences]
            
        reward = torch.cat(reward_list)
        ret = torch.cat(return_list)
        reward_mean, reward_var, _ = get_global_statistics(reward)
        return_mean, return_var, _ = get_global_statistics(ret)
        reward_std = torch.sqrt(reward_var)
        return_std = torch.sqrt(return_var)
        
        # Token-level PPO & Token-mix-sample RM
        if self.ppo_granularity is ActorGranularity.token and self.rm_granularity is RewardModelGranularity.token_mix_sample:
            sample_reward = torch.cat(sample_level_reward_list)
            sample_ret = torch.cat(sample_level_return_list)
            sample_reward_mean, sample_reward_var, _ = get_global_statistics(sample_reward)
            sample_return_mean, sample_return_var, _ = get_global_statistics(sample_ret)
            sample_reward_std = torch.sqrt(sample_reward_var)
            sample_return_std = torch.sqrt(sample_return_var)
        
        if is_guide:
            reward_mean, reward_std = self.experience_maker.recover_reward(reward_mean, reward_std, True)
            self.logger.log_metrics({
                "reward_guide_epoch_mean": reward_mean.item(),
                "reward_guide_epoch_std": reward_std.item(),
                "return_guide_epoch_mean": return_mean.item(),
                "return_guide_epoch_std": return_std.item()
            }, step=self.global_steps, metrics_group='reward')
            
            # Token-level PPO & Token-mix-sample RM
            if self.ppo_granularity is ActorGranularity.token and self.rm_granularity is RewardModelGranularity.token_mix_sample:
                sample_reward_mean, sample_reward_std = self.experience_maker.recover_reward(sample_reward_mean, sample_reward_std, False)
                self.logger.log_metrics({
                    "sample_level_reward_guide_epoch_mean": sample_reward_mean.item(),
                    "sample_level_reward_guide_epoch_std": sample_reward_std.item(),
                    "sample_level_return_guide_epoch_mean": sample_return_mean.item(),
                    "sample_level_return_guide_epoch_std": sample_return_std.item()
                }, step=self.global_steps, metrics_group='reward')
        else:
            reward_mean, reward_std = self.experience_maker.recover_reward(reward_mean, reward_std, True)
            self.logger.log_metrics({
                "reward_epoch_mean": reward_mean.item(),
                "reward_epoch_std": reward_std.item(),
                "return_epoch_mean": return_mean.item(),
                "return_epoch_std": return_std.item()
            }, step=self.global_steps, metrics_group='reward')
            
            # Token-level PPO & Token-mix-sample RM
            if self.ppo_granularity is ActorGranularity.token and self.rm_granularity is RewardModelGranularity.token_mix_sample:
                sample_reward_mean, sample_reward_std = self.experience_maker.recover_reward(sample_reward_mean, sample_reward_std, False)
                self.logger.log_metrics({
                    "sample_level_reward_epoch_mean": sample_reward_mean.item(),
                    "sample_level_reward_epoch_std": sample_reward_std.item(),
                    "sample_level_return_epoch_mean": sample_return_mean.item(),
                    "sample_level_return_epoch_std": sample_return_std.item()
                }, step=self.global_steps, metrics_group='reward')

        self.experience_maker.logging(step=self.global_steps)
        
        return
