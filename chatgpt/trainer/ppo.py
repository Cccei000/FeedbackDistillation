import gc
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from fengshen_inner.models.megatron import mpu
from torch.optim import Optimizer
from tqdm import tqdm
from copy import deepcopy

from chatgpt.experience_maker import Experience, LocalInferExperienceMaker
from chatgpt.logger.base import Logger
from chatgpt.nn import Actor, Critic, EntropyLoss, PolicyLoss, ValueLoss
from chatgpt.nn.utils import (compute_approx_kl, get_global_statistics,
                              get_reward_by_mask, masked_mean,
                              zero_pad_sequences)
from chatgpt.pipeline.config import ActorGranularity, RewardModelGranularity
from chatgpt.replay_buffer import ReplayBuffer
from chatgpt.utils import (ACTOR_TRAIN, ALL_MODELS, CRITIC_TRAIN, TRAIN_MODELS,
                           CostTimer, DeepspeedFlopsTimerGroup,
                           FlopsTimerGroup, is_rank_0, logging_rank_0)

from transformers import PreTrainedTokenizer
from .base import Trainer, AdaptiveKLController, FixedKLController
from .callbacks import Callback

START_BACKWARD = 1
SKIP_BACKWARD = 0


class PPOTrainer(Trainer):

    def __init__(self,
                 actor: Actor,
                 critic: Critic,
                 actor_optim: Optimizer,
                 critic_optim: Optimizer,
                 actor_lr_scheduler,
                 critic_lr_scheduler,
                 experience_maker: LocalInferExperienceMaker,
                 replay_buffer: ReplayBuffer,
                 setup_dataloader_func: Callable,
                 logger: Logger,
                 ckpt_saving_func: Callable,
                 rm_granularity: RewardModelGranularity,
                 ppo_granularity: ActorGranularity,
                 tokenizer: PreTrainedTokenizer,
                 constrain_actor: Optional[Actor] = None,
                 eps_clip: float = 0.2,
                 value_clip: float = 0.4,
                 drop_approx_kl: float = 1e-3,
                 experience_batch_size: int = 512,
                 max_epochs: int = 1,
                 sample_replay_buffer: bool = False,
                 entropy_loss_coef: float = 0.01,
                 entropy_loss_decay_rate: float = 1,
                 constrain_actor_kl_coef: float = 0.01,
                 target_constrain_actor_kl: Optional[float] = None,
                 kl_adaptor_horizon: Optional[float] = None,
                 update_constrain_actor_interval: int = 5,
                 clip_grad: bool = False,
                 callbacks: List[Callback] = [],
                 flops_timers: Optional[FlopsTimerGroup]=None,
                 deepspeed_flops_timers: Optional[DeepspeedFlopsTimerGroup]=None,
                 **kwargs) -> None:
        """Trainer for PPO algorithm.

        Args:
            actor (Actor): Actor模型
            critic (Critic): Critic模型
            actor_optim (Optimizer): Actor的优化器
            critic_optim (Optimizer): Critic的优化器
            actor_lr_scheduler (_type_): Actor的LR Scheduler
            critic_lr_scheduler (_type_): Critic的LR Scheduler
            experience_maker (LocalInferExperienceMaker): 经验池采样器
            replay_buffer (ReplayBuffer): 经验池
            setup_dataloader_func (Callable): DataLoader构造方法
            logger (Logger): Logger
            ckpt_saving_func (Callable): 模型Checkpoint保存方法
            rm_granularity (RewardModelGranularity): RM训练时采用的粒度
            ppo_granularity (ActorGranularity): PPO流程的粒度
            tokenizer (PreTrainedTokenizer): Actor模型的Tokenizer
            constrain_actor (Optional[Actor], optional): 用于约束的Actor备份模型. Defaults to None.
            eps_clip (float, optional): eps clipping采用的上下限阈值. Defaults to 0.2.
            value_clip (float, optional): value clipping采用的上下限阈值. Defaults to 0.4.
            drop_approx_kl (float, optional): training step skipping采用的阈值. Defaults to 1e-3.
            experience_batch_size (int, optional): 每轮采样经验池的大小. Defaults to 512.
            max_epochs (int, optional): 整体的迭代轮次（每轮包括经验池采样和模型训练）. Defaults to 1.
            sample_replay_buffer (bool, optional): 每次训练是否仅从经验池中采样部分样本. Defaults to False.
            entropy_loss_coef (float, optional): 动作熵激励系数. Defaults to 0.01.
            entropy_loss_decay_rate (float, optional): 动作熵激励系数的衰减系数. Defaults to 1.
            constrain_actor_kl_coef (float, optional): Actor约束模型的惩罚系数. Defaults to 0.01.
            target_constrain_actor_kl (Optional[float], optional): 与Actor备份模型Approx kl的预期值，设置为None时不启动AdaptiveKLControlller，否则启动之. Defaults to None.
            kl_adaptor_horizon (Optional[float], optional): AdaptiveKLControlller用于控制constrain_actor_kl_coef变化的速率，越小变化越快. Defaults to None.
            update_constrain_actor_interval (int, optional): 更新Actor备份模型的频率. Defaults to 5.
            clip_grad (bool, optional): 是否实施gradient clipping. Defaults to False.
            callbacks (List[Callback], optional): 预留的callbacks. Defaults to [].
            flops_timers (Optional[FlopsTimerGroup], optional): flops_timers. Defaults to None.
            deepspeed_flops_timers (Optional[DeepspeedFlopsTimerGroup], optional): Deepspeed提供的flops_timers. Defaults to None.
        """      
        super().__init__(experience_maker=experience_maker,
                         replay_buffer=replay_buffer,
                         experience_batch_size=experience_batch_size,
                         setup_dataloader_func=setup_dataloader_func,
                         max_epochs=max_epochs,
                         tokenizer=tokenizer,
                         sample_replay_buffer=sample_replay_buffer,
                         callbacks=callbacks,
                         logger=logger)
        
        ### Model ###
        self.actor = actor
        self.critic = critic
        self.constrain_actor = constrain_actor
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_lr_scheduler = actor_lr_scheduler
        self.critic_lr_scheduler = critic_lr_scheduler
        
        ### Granularity ###
        self.ppo_granularity = ppo_granularity
        self.rm_granularity = rm_granularity
        
        ### Loss & Training ###
        enable_token_level_loss =self.ppo_granularity is ActorGranularity.token
        self.actor_loss_fn = PolicyLoss(eps_clip, token_level_mean=enable_token_level_loss)
        self.entropy_loss_fn = EntropyLoss()
        self.critic_loss_fn = ValueLoss(value_clip, token_level_mean=enable_token_level_loss)
        self.entropy_loss_coef = entropy_loss_coef
        self.entropy_loss_decay_rate = entropy_loss_decay_rate
        self.drop_approx_kl = drop_approx_kl
        self.clip_grad = clip_grad
        
        self.enable_constrain_actor:bool = self.constrain_actor is not None     # 是否对 pg_loss 启用 kl 约束
        self.enable_constrain_actor_updating:bool = update_constrain_actor_interval > 0  # 是否固定 kl 约束 actor
        self.update_constrain_actor_interval = update_constrain_actor_interval    # kl 约束 actor更新频率
        
        if target_constrain_actor_kl is not None:
            self.constrain_actor_kl_coef = AdaptiveKLController(
                kl_coef=constrain_actor_kl_coef,
                target=target_constrain_actor_kl,
                horizon=kl_adaptor_horizon if kl_adaptor_horizon is not None else experience_batch_size
            )
        else:
            self.constrain_actor_kl_coef = FixedKLController(kl_coef=constrain_actor_kl_coef)                  # kl 约束在 pg_loss 的比重

        ### Loggers ###
        self.flops_timers = flops_timers
        self.deepspeed_flops_timers = deepspeed_flops_timers
        
        ### Others ###
        self.pad_token_id = self.tokenizer.pad_token_id
        self.ckpt_saving_func = ckpt_saving_func
        self.sync_info = torch.tensor([0], dtype=torch.int, device=self.actor.device)
        
        if self.enable_constrain_actor and self.constrain_actor is None:
            logging_rank_0(f"Warning: Missing 'constrain_model' when enabling 'constrain_actor'. Disable it.", "debug")
            self.enable_constrain_actor = False
        if self.enable_constrain_actor:
            if self.enable_constrain_actor_updating:
                logging_rank_0(f"Enable Constrain Actor with updating it every {self.update_constrain_actor_interval} ep. And its coef is {self.constrain_actor_kl_coef}.", "debug")
            else:
                logging_rank_0(f"Enable Constrain Actor which is fixed and same as initial model.", "debug")
        else:
            logging_rank_0(f"Disable Constrain Actor.", "debug")
        
        ### unused param ###
        for key in kwargs:
            logging_rank_0(f"Deprecation Warning: Attribute '{key}' is not used in 'PPOTrainer'.", "debug")
            
        return   
    
    def _actor_training_step(self, experience: Experience) -> Dict[str, float]:
        """训练一下Actor

        Args:
            experience (Experience): _description_

        Returns:
            Dict[str, float]: _description_
        """
        
        num_actions = experience.action_mask.size(1)
        
        ### Actor 训练 ###
        self._on_forward_start(ACTOR_TRAIN)
        action_log_probs, logits = self.actor(
            experience.sequences, num_actions, attention_mask=experience.attention_mask, return_logits=True)
        
        approx_kl = compute_approx_kl(
            action_log_probs, experience.action_log_probs, action_mask=experience.action_mask, return_mean=True
        )
        
        actor_loss, ratio = self.actor_loss_fn(action_log_probs,
                                        experience.action_log_probs,
                                        experience.advantages,
                                        action_mask=experience.action_mask,
                                        return_ratio=True)
        entropy = self.entropy_loss_fn(logits, experience.action_mask)  # 计算动作熵
        actor_forward_time = CostTimer.get_time()

        actor_total_loss = actor_loss - self.entropy_loss_coef * entropy
        
        # 使用 constrain actor 进行 kl 约束
        if self.enable_constrain_actor:
            action_log_probs_initial_model_update, _ = self.constrain_actor(
                experience.sequences, num_actions, attention_mask=experience.attention_mask, return_logits=True)
            approx_kl_with_constrain_actor = compute_approx_kl(
                action_log_probs, action_log_probs_initial_model_update, action_mask=experience.action_mask, return_mean=True
            )
            n_steps = approx_kl_with_constrain_actor.shape[0]
            approx_kl_with_constrain_actor = torch.mean(approx_kl_with_constrain_actor)
            self.constrain_actor_kl_coef.update(current=approx_kl_with_constrain_actor.item(), n_steps=n_steps)
            actor_total_loss = actor_total_loss + self.constrain_actor_kl_coef.value * approx_kl_with_constrain_actor
        else:
            approx_kl_with_constrain_actor = None
        
        # 丢弃kl过大的样本
        drop = torch.abs(approx_kl).mean().item() >= self.drop_approx_kl
        
        self._on_actor_backward_start(do_backward=not drop)
        
        actor_backward_time = 0.0
        if not drop and self.do_actor_backward:
            with CostTimer():
                self.actor.backward(actor_total_loss)
                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor.step()
                self.actor.zero_grad()
            actor_backward_time = CostTimer.get_time()
        elif not drop and not self.do_actor_backward:
            logging_rank_0(f"Skip backward because of other dp rank.", "debug")
        else:
            logging_rank_0(f"Skip backward because of approx kl ({torch.abs(approx_kl).mean().item()}).", "debug")
        
        self._on_forward_end(
            model_name=ACTOR_TRAIN, 
            drop=not self.do_actor_backward,
            batch_size=experience.sequences.shape[0], 
            seq_length=experience.sequences.shape[1], 
            time=actor_forward_time + actor_backward_time
        )
        logging_rank_0(f"ACTOR:{actor_forward_time} + {actor_backward_time}", "debug")
        rsp = {
            'actor_loss': actor_loss.item(),
            'ratio': ratio.mean().item(),
            'entropy': entropy.item(),
            'kl_step':approx_kl.mean().item(),
        }
        if approx_kl_with_constrain_actor is not None:
            rsp["approx_kl_with_constrain_actor"] = approx_kl_with_constrain_actor.item()
            rsp["approx_kl_coef_with_constrain"] = self.constrain_actor_kl_coef.value
        
        return rsp
    
    def _critic_training_step(self, experience: Experience) -> Dict[str, float]:
        """训练一下Critic

        Args:
            experience (Experience): _description_

        Returns:
            Dict[str, float]: _description_
        """
         ### Critic 训练 ###
        self._on_forward_start(CRITIC_TRAIN)
        values = self.critic(experience.sequences,
                             action_mask=experience.action_mask,
                             attention_mask=experience.attention_mask)
        critic_forward_time = CostTimer.get_time()
        with CostTimer():
            critic_loss = self.critic_loss_fn(values,
                                            experience.values,
                                            experience.reward,
                                            action_mask=experience.action_mask)

            self.critic.backward(critic_loss)
            # if self.clip_grad:
            #     torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic.step()
            self.critic.zero_grad()
        critic_backward_time = CostTimer.get_time()
        
        logging_rank_0(f"CRITIC:{critic_forward_time} + {critic_backward_time}", "debug")
        self._on_forward_end(CRITIC_TRAIN, batch_size=experience.sequences.shape[0], seq_length=experience.sequences.shape[1], time=critic_forward_time +critic_backward_time)
        
        return {
            'critic_loss': critic_loss.item(),
        }
        
    def training_step(self, experience: Experience) -> Dict[str, float]:
        """Training step

        Args:
            experience (Experience): 本步训练使用的样本

        Returns:
            Dict[str, float]: 本部训练的信息
        """    
        
        rsp = {}    

        actor_rsp = self._actor_training_step(experience=experience)
        torch.cuda.empty_cache()
        gc.collect()
        critic_rsp = self._critic_training_step(experience=experience)
        torch.cuda.empty_cache()
        gc.collect()
        
        rsp.update(actor_rsp)
        rsp.update(critic_rsp)
       
        with CostTimer():
        
            ### 整合 logger 需要记录的信息 ###
            # Token-level PPO
            if self.ppo_granularity is ActorGranularity.token:
                # Token-level RM
                if self.rm_granularity is RewardModelGranularity.token:
                    return_mean = masked_mean(experience.reward, experience.action_mask).mean()
                    reward_mean = masked_mean(experience.origin_reward, experience.action_mask).mean()
                # Sample-level RM
                elif self.rm_granularity is RewardModelGranularity.sample:
                    return_mean = get_reward_by_mask(experience.reward, experience.action_mask).mean()
                    reward_mean = get_reward_by_mask(experience.origin_reward, experience.action_mask).mean()
                # Token-mix-sample RM
                else:
                    return_mean = masked_mean(experience.reward, experience.action_mask).mean()
                    reward_mean = masked_mean(experience.origin_reward, experience.action_mask)
                    sample_return_mean = get_reward_by_mask(experience.reward, experience.action_mask).mean()
                    sample_reward_mean = get_reward_by_mask(experience.origin_reward, experience.action_mask)
                    # reward_mean = reward_mean - sample_return_mean  ## 分离
                    reward_mean = reward_mean.mean()
                    sample_reward_mean = sample_reward_mean.mean()
            # Sample-level PPO || Step-level PPO
            else:
                return_mean = experience.reward.mean()
                reward_mean = experience.origin_reward.mean()

            self.logger.log_metrics(rsp, step=self.global_steps, metrics_group='train')
            reward_mean, _ = self.experience_maker.recover_reward(reward_mean, None, is_token=True)
            self.logger.log_metrics({
                "return_step": return_mean.item(),
                "reward_step": reward_mean.item(),
            }, step=self.global_steps, metrics_group='reward')
            self.logger.log_metrics({"actor_lr": self.actor_lr_scheduler.get_last_lr()[0], "critic_lr": self.critic_lr_scheduler.get_last_lr()[0], "entropy_coef": self.entropy_loss_coef}, step=self.global_steps, metrics_group='lr')
            
            # Token-level PPO & Token-mix-sample RM 记录 mixed sample-level 奖励
            if self.ppo_granularity is ActorGranularity.token and self.rm_granularity is RewardModelGranularity.token_mix_sample:
                sample_reward_mean, _ = self.experience_maker.recover_reward(sample_reward_mean, None, False)
                self.logger.log_metrics({
                    "sample_level_return_step": sample_return_mean.item(),
                    "sample_level_reward_step": sample_reward_mean.item(),
                }, step=self.global_steps, metrics_group='reward')
        
        logging_rank_0(f"LOG:{CostTimer.get_time()}", "debug")
        gc.collect()
        torch.cuda.empty_cache()
        return rsp
    
    def validation_step(self, experience: Experience) -> Dict[str, Any]:
        return
    
    def fit(self,
            prompts,
            val_prompts:Dict[str,List[str]]=None,
            seed: int = 42,
            num_episodes: int = 50000,
            max_timesteps: int = 500,
            update_timesteps: int = 5000,
            val_check_interval:int=2,
            val_saving_func:Optional[Callable] = None) -> None:
        """_summary_

        Args:
            prompts (_type_): 用于训练的所有prompt
            val_prompts (Dict[str,List[str]], optional): 用于验证的prompt. Defaults to None.
            seed (int, optional): 随机种子. Defaults to 42.
            num_episodes (int, optional): _description_. Defaults to 50000.
            max_timesteps (int, optional): _description_. Defaults to 500.
            update_timesteps (int, optional): _description_. Defaults to 5000.
            val_check_interval (int, optional): _description_. Defaults to 2.
            val_saving_func (Optional[Callable], optional): 验证频率，每过几个episode验证. Defaults to None.

        Returns:
            _type_: _description_
        """        
        
        # 不启动验证集，运行base的fit
        if val_prompts is None:
            return super().fit(prompts=prompts, num_episodes=num_episodes, max_timesteps=max_timesteps, update_timesteps=update_timesteps)
        
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
                self._on_episode_end(episode)
                continue
            
            self.timers('episode').start()
            self._on_episode_start(episode)
            # 采样训练
            for timestep in tqdm(range(max_timesteps),
                                desc=f'Episode [{episode+1}/{num_episodes}]',
                                disable=not is_rank_0()):
                time += 1
                rand_prompts = self._sample_prompts(prompts, seed+timestep+episode)    # 采样训练的prompt
                if self.tokenizer is not None:
                    inputs = self.tokenizer.batch_encode_plus(rand_prompts)["input_ids"]
                else:
                    inputs = rand_prompts
                self.timers('make_experience').start()
                self._on_make_experience_start()
                self.make_exp_steps += 1
                experiences = self._make_experience(inputs)
                #debug
                self._on_make_experience_end(experiences)
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
                gc.collect()
                torch.cuda.empty_cache()
            self._on_episode_end(episode)
            self.timers('episode').stop()
            self.timers.write(['episode'], self.global_steps, reset=True, metrics_group='timers')
        self._on_fit_end()

    
    def _make_experience(self, inputs: List[List[int]], is_val:bool=False) -> List[Experience]:
        """采样经验池

        Args:
            inputs (List[List[int]]): 一批query的input_ids
            is_val (bool, optional): 是否是验证集. Defaults to False.

        Returns:
            List[Experience]: _description_
        """
        
        input_ids = [torch.tensor(ids) for ids in inputs] # 转换为torch.Tensor
        
        # right padding
        inputs = zero_pad_sequences(input_ids, side="right", # side="left",
                                    padding_value=self.pad_token_id)
        
        # 调用experience maker，采样经验池
        exp, kl_ref = self.experience_maker.make_experience(inputs, return_kl=True, step=self.global_steps, is_val_make=is_val)
        # 计算本批经验池的Approx KL
        kl_ref_mean, _, _ = get_global_statistics(kl_ref)
        if is_val:
            self.logger.log_metrics({"kl_ref_val": kl_ref_mean.item()}, step=self.global_steps, metrics_group='experience')
        else:
            self.logger.log_metrics({"kl_ref": kl_ref_mean.item()}, step=self.global_steps, metrics_group='experience')
        return exp
    
    
    def _on_make_experience_start(self) -> None:
        # 在开始生成exp的时候，把grad全部设为None，不然grad依旧占用显存
        self.actor.zero_grad()
        self.critic.zero_grad()
        return super()._on_make_experience_start()

    def _on_make_experience_end(self, experiences: List[Experience]) -> None:
        """采样完经验池后记录奖励值信息

        Args:
            experiences (List[Experience]): _description_

        Returns:
            _type_: _description_
        """
        
        assert len(experiences) > 0

        # 计算本批experience的奖励值信息
        # Token-level PPO
        if self.ppo_granularity is ActorGranularity.token:
            return_list = []
            reward_list = []
            sample_level_return_list = []
            sample_level_reward_list = []
            # Sample-level RM
            if self.rm_granularity is RewardModelGranularity.token:
                for exp in experiences:
                    last_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in exp.attention_mask[:, 1:]], device=exp.reward.device, dtype=torch.long)
                    reward_list.append(exp.origin_reward[torch.arange(0,exp.origin_reward.shape[0], device=exp.origin_reward.device), last_index])
                    return_list.append(exp.reward[torch.arange(0,exp.reward.shape[0], device=exp.reward.device), last_index])
            ## Token-mix-sample RM
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
            ## 计算 mixed sample-level 奖励
            if self.mixed_sample_level_reward:
                return_list = []
                reward_list = []
                for exp in experiences:
                    last_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in exp.attention_mask[:, 1:]], device=exp.reward.device, dtype=torch.long)
                    sample_level_reward_list.append(exp.origin_reward[torch.arange(0,exp.origin_reward.shape[0], device=exp.origin_reward.device), last_index])
                    sample_level_return_list.append(exp.reward[torch.arange(0,exp.reward.shape[0], device=exp.reward.device), last_index])
        # Sample-level PPO || Step-level PPO
        else:
            reward_list = [exp.origin_reward for exp in experiences]
            return_list = [exp.reward for exp in experiences]
            
        reward = torch.cat(reward_list)
        ret = torch.cat(return_list)

        reward_mean, reward_var, _ = get_global_statistics(reward)
        return_mean, return_var, _ = get_global_statistics(ret)
        reward_std = torch.sqrt(reward_var)
        return_std = torch.sqrt(return_var)
        
        # Token-level PPO || Token-mix-sample PPO
        if self.ppo_granularity is ActorGranularity.token and self.rm_granularity is RewardModelGranularity.token_mix_sample:
            sample_reward = torch.cat(sample_level_reward_list)
            sample_ret = torch.cat(sample_level_return_list)
            sample_reward_mean, sample_reward_var, _ = get_global_statistics(sample_reward)
            sample_return_mean, sample_return_var, _ = get_global_statistics(sample_ret)
            sample_reward_std = torch.sqrt(sample_reward_var)
            sample_return_std = torch.sqrt(sample_return_var)
            
            reward_mean, reward_std = self.experience_maker.recover_reward(reward_mean, reward_std, True)
            sample_reward_mean, sample_reward_std = self.experience_maker.recover_reward(sample_reward_mean, sample_reward_std, False)
            
            self.logger.log_metrics({
                "reward_epoch_mean": reward_mean.item(),
                "reward_epoch_std": reward_std.item(),
                "return_epoch_mean": return_mean.item(),
                "return_epoch_std": return_std.item(),
                "sample_level_reward_epoch_mean": sample_reward_mean.item(),
                "sample_level_reward_epoch_std": sample_reward_std.item(),
                "sample_level_return_epoch_mean": sample_return_mean.item(),
                "sample_level_return_epoch_std": sample_return_std.item(),
            }, step=self.global_steps, metrics_group='reward')
        
        else:
            self.logger.log_metrics({
                "reward_epoch_mean": reward_mean.item(),
                "reward_epoch_std": reward_std.item(),
                "return_epoch_mean": return_mean.item(),
                "return_epoch_std": return_std.item(),
            }, step=self.global_steps, metrics_group='reward')
        
        self.experience_maker.logging(step=self.global_steps)

        return super()._on_make_experience_end(experiences)

    def _on_learn_epoch_start(self, epoch: int) -> None:
        return super()._on_learn_epoch_start(epoch)

    def _on_learn_epoch_end(self, epoch: int) -> None:
        self.entropy_loss_coef = max(0.01, self.entropy_loss_coef * self.entropy_loss_decay_rate)
        return super()._on_learn_epoch_end(epoch)

    def _on_learn_batch_start(self) -> None:
        return super()._on_learn_batch_start()

    def _on_learn_batch_end(self, metrics: dict, experience: Experience) -> None:
        return super()._on_learn_batch_end(metrics, experience)
    
    def _on_episode_start(self, episode: int) -> None:
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()
        return super()._on_episode_start(episode)

    def _on_episode_end(self, episode: int) -> None:
        if self.enable_constrain_actor:
            if episode % self.update_constrain_actor_interval == 0:
                if self.enable_constrain_actor_updating:
                    self.constrain_actor = deepcopy(self.actor.module)
                    logging_rank_0(f'Constrain Actor is updated. Current episode is {episode}.', 'debug')
                for param in self.constrain_actor.named_parameters():
                    param[1].requires_grad = False
            self.constrain_actor.to('cpu')
        return super()._on_episode_end(episode)
    
    def _log_val_metrics(self, experiences: List[Experience], val_prompts:Dict[str,list]) -> None:
        """采样完验证集的经验池后记录奖励值信息

        Args:
            experiences (List[Experience]): _description_
            val_prompts (Dict[str,list]): _description_
        """
        assert len(experiences) > 0
        
        # Token-level PPO
        if self.ppo_granularity is ActorGranularity.token:
            reward_list, return_list = [], []
            sample_level_reward_list, sample_level_return_list = [], []
            # Sample-level RM
            if self.rm_granularity is RewardModelGranularity.sample:
                for exp in experiences:
                    last_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in exp.attention_mask[:, 1:]], device=exp.reward.device, dtype=torch.long)
                    return_list.append(exp.reward[torch.arange(0,exp.reward.shape[0], device=exp.reward.device), last_index])
                    reward_list.append(exp.origin_reward[torch.arange(0,exp.origin_reward.shape[0], device=exp.origin_reward.device), last_index])
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
                    return_list.append(masked_mean(exp.reward,exp.action_mask))
                    reward_list.append(masked_mean(exp.origin_reward, exp.action_mask))
        # Sample-level PPO || Step-level PPO
        else:
            return_list = [exp.reward for exp in experiences]
            reward_list = [exp.origin_reward for exp in experiences]
            
        reward = torch.cat(reward_list)
        ret = torch.cat(return_list)
        
        # Token-level PPO & Token-mix-sample PPO
        if self.ppo_granularity is ActorGranularity.token and self.rm_granularity is RewardModelGranularity.token_mix_sample:
            sample_reward = torch.cat(sample_level_reward_list)
            sample_ret = torch.cat(sample_level_return_list)
        
        base_idx = 0
        for task_name, prompts in val_prompts.items():
            prompt_num = len(prompts)
            task_rewards = reward[base_idx:base_idx+prompt_num]
            task_returns = ret[base_idx:base_idx+prompt_num]
            reward_mean, reward_var, _ = get_global_statistics(task_rewards)
            return_mean, return_var, _ = get_global_statistics(task_returns)
            reward_std = torch.sqrt(reward_var)
            return_std = torch.sqrt(return_var)
            
            reward_mean, reward_std = self.experience_maker.recover_reward(reward_mean, reward_std, True)
            
            self.logger.log_metrics({task_name: reward_mean.item()}, step=self.global_steps, metrics_group='val_reward_mean')
            self.logger.log_metrics({task_name: reward_std.item()}, step=self.global_steps, metrics_group='val_reward_std')
            self.logger.log_metrics({task_name: return_mean.item()}, step=self.global_steps, metrics_group='val_return_mean')
            self.logger.log_metrics({task_name: return_std.item()}, step=self.global_steps, metrics_group='val_return_std')
            
            # Token-level PPO & Token-mix-sample RM：记录sample-level奖励
            if self.ppo_granularity is ActorGranularity.token and self.rm_granularity is RewardModelGranularity.token_mix_sample:
                task_rewards = sample_reward[base_idx:base_idx+prompt_num]
                task_returns = sample_ret[base_idx:base_idx+prompt_num]
                reward_mean, reward_var, _ = get_global_statistics(task_rewards)
                return_mean, return_var, _ = get_global_statistics(task_returns)
                reward_std = torch.sqrt(reward_var)
                return_std = torch.sqrt(return_var)
                
                reward_mean, reward_std = self.experience_maker.recover_reward(reward_mean, reward_std, False)
                
                self.logger.log_metrics({task_name: reward_mean.item()}, step=self.global_steps, metrics_group='val_sample_level_reward_mean')
                self.logger.log_metrics({task_name: reward_std.item()}, step=self.global_steps, metrics_group='val_sample_level_reward_std')
                self.logger.log_metrics({task_name: return_mean.item()}, step=self.global_steps, metrics_group='val_sample_level_return_mean')
                self.logger.log_metrics({task_name: return_std.item()}, step=self.global_steps, metrics_group='val_sample_level_return_std')
            
            base_idx += prompt_num
        
            
        reward_mean, reward_var, _ = get_global_statistics(reward)
        return_mean, return_var, _ = get_global_statistics(ret)
        reward_std = torch.sqrt(reward_var)
        return_std = torch.sqrt(return_var)
        reward_mean, reward_std = self.experience_maker.recover_reward(reward_mean, reward_std, True)
        self.logger.log_metrics({"all": reward_mean.item()}, step=self.global_steps, metrics_group='val_reward_mean')
        self.logger.log_metrics({"all": reward_std.item()}, step=self.global_steps, metrics_group='val_reward_std')
        self.logger.log_metrics({"all": return_mean.item()}, step=self.global_steps, metrics_group='val_return_mean')
        self.logger.log_metrics({"all": return_std.item()}, step=self.global_steps, metrics_group='val_return_std')

        # Token-level PPO & Token-mix-sample RM：记录sample-level奖励
        if self.ppo_granularity is ActorGranularity.token and self.rm_granularity is RewardModelGranularity.token_mix_sample:
            reward_mean, reward_var, _ = get_global_statistics(sample_reward)
            return_mean, return_var, _ = get_global_statistics(sample_ret)
            reward_std = torch.sqrt(reward_var)
            return_std = torch.sqrt(return_var)
            reward_mean, reward_std = self.experience_maker.recover_reward(reward_mean, reward_std, False)
            self.logger.log_metrics({"all": reward_mean.item()}, step=self.global_steps, metrics_group='val_sample_level_reward_mean')
            self.logger.log_metrics({"all": reward_std.item()}, step=self.global_steps, metrics_group='val_sample_level_reward_std')
            self.logger.log_metrics({"all": return_mean.item()}, step=self.global_steps, metrics_group='val_sample_level_return_mean')
            self.logger.log_metrics({"all": return_std.item()}, step=self.global_steps, metrics_group='val_sample_level_return_std')
        return
    
    def _on_actor_backward_start(self, do_backward:bool=True) -> None:
        """在Actor反向之前同步不同dp rank的信号，确认是否进行反向

        Args:
            do_backward (bool, optional): 本dp rank是否准备进行反向. Defaults to True.
        """
        
        self.sync_info[0] = START_BACKWARD if do_backward else SKIP_BACKWARD
        
        if dist.is_initialized():
            ### gather
            dp_world_size = mpu.get_data_parallel_world_size()
            ### dp1 不需要同步
            if dp_world_size == 1:
                return
            synced_info = [torch.zeros_like(self.sync_info, device=self.sync_info.device) for _ in range(dp_world_size)]
            dist.all_gather(synced_info, self.sync_info, group=mpu.get_data_parallel_group())
            for info in synced_info:
                if info[0] == SKIP_BACKWARD:
                    self.sync_info[0] = SKIP_BACKWARD
                    break
                
        return

    @property
    def do_actor_backward(self) -> bool:
        """判断是否满足Actor反向传播的条件

        Returns:
            bool: 是否满足Actor反向传播的条件
        """
        return self.sync_info[0] == START_BACKWARD

    def _on_forward_start(self, model_name:str) -> None:
        """在模型开始前向之前进行的操作

        Args:
            model_name (str): 即将前向的模型名（定义于chatgpt.utils.tflops.py）
        """
        # 记录生成的计算量
        if self.deepspeed_flops_timers is not None:
            self.deepspeed_flops_timers.start_profile(model_name)
        return

    def _on_forward_end(self, model_name:str, drop:bool=False, **kwargs) -> None:
        """在模型结束前向之后进行的操作

        Args:
            model_name (str): 已经前向的模型名（定义于chatgpt.utils.tflops.py）
            drop (bool, optional): _description_. Defaults to False.
        """
        
        seq_length = kwargs.get("seq_length", None)
        batch_size = kwargs.get("batch_size", None)
        batch_info = kwargs.get("batch_info", None)
        iter_time_s = kwargs.get("time", None)
        
        # logging_rank_0(f"{model_name}: {iter_time_s}", "debug")
        
        # 记录生成的计算量
        if self.deepspeed_flops_timers is not None:
            self.deepspeed_flops_timers.stop_profile(model_name)
        if self.flops_timers is not None and not drop:
            if iter_time_s is not None and batch_info is not None:
                self.flops_timers.update_timer(model_name=model_name, iter_time_s=iter_time_s)
                self.flops_timers.update_calculation(model_name=model_name, batch_info=batch_info)
            elif iter_time_s is not None and seq_length is not None and batch_size is not None:
                self.flops_timers.update_timer(model_name=model_name, iter_time_s=iter_time_s)
                self.flops_timers.update_calculation(model_name=model_name, batch_size=batch_size, seq_length=seq_length)
        
        return
    
    def _on_learn_start(self) -> None:
        """训练阶段开始之前的操作
        """
        
        # 重置所有flops记录器
        if self.flops_timers is not None:
            self.flops_timers.reset(ACTOR_TRAIN)
            self.flops_timers.reset(CRITIC_TRAIN)
        if self.deepspeed_flops_timers is not None:
            self.deepspeed_flops_timers.end_profile(ACTOR_TRAIN)
            self.deepspeed_flops_timers.end_profile(CRITIC_TRAIN)
            
        if self.enable_constrain_actor:
            self.constrain_actor.cuda()
        return 
            
    def _on_profile_flops(self, step:int) -> None:
        """更新flops记录器

        Args:
            step (int): global step
        """
        
        if self.flops_timers is not None:
            flops_dict = self.flops_timers.get_flops(TRAIN_MODELS)
            log_info = {
                f"{name}": flops
                for name, flops in flops_dict.items()
            }
            log_info["train_all"] = self.flops_timers.get_avg_flops(model_name=TRAIN_MODELS)
            log_info["all"] = self.flops_timers.get_avg_flops(model_name=ALL_MODELS)
            self.logger.log_metrics(log_info, step=self.global_steps, metrics_group='flops')
            
        if self.deepspeed_flops_timers is not None:
            flops_dict = self.deepspeed_flops_timers.get_flops(TRAIN_MODELS)
            log_info = {
                f"{name}": flops
                for name, flops in flops_dict.items()
            }
            log_info["train_all"] = self.deepspeed_flops_timers.get_avg_flops(model_name=TRAIN_MODELS)
            log_info["all"] = self.deepspeed_flops_timers.get_avg_flops(model_name=ALL_MODELS)
            self.logger.log_metrics(log_info, step=step, metrics_group='flops_deepspeed')

        return
