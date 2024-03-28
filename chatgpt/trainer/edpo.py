## Direct Preference Optimizaiton(DPO)
## Rafailov et. al. Direct Preference Optimization: Your Language Model is Secretly a Reward Model, 2023

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch.nn as nn
import torch.distributed as dist
from chatgpt.experience_maker import PreferenceExperience, PreferenceExperienceMaker
from chatgpt.replay_buffer import PreferenceReplayBuffer
from chatgpt.logger.base import Logger
from chatgpt.utils import is_rank_0
from chatgpt.nn import Actor, PreferenceLoss
from chatgpt.nn.utils import zero_pad_sequences, compute_approx_kl, get_global_statistics, masked_mean
from torch.optim import Optimizer
from tqdm import tqdm
import torch
import gc

from collections import defaultdict

from .base import Trainer
from .callbacks import Callback
from chatgpt.utils import logging_rank_0
from fengshen_inner.models.megatron import mpu

import time



class EDPOTrainer(Trainer):
    """
        Trainer for EDPO algorithm.

    Args:
        actor (Actor): the actor model in ppo algorithm
        actor_optim (Optimizer): the optimizer to use for actor model
        buffer_cpu_offload (bool, defaults to True): whether to offload replay buffer to cpu
        eps_clip (float, defaults to 0.2): the clip coefficient of policy loss
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        tokenier (Callable, optional): the tokenizer to use for tokenizing the input
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    """

    def __init__(self,
                 actor: Actor,
                 actor_optim: Optimizer,
                 actor_lr_scheduler,
                 experience_maker: PreferenceExperienceMaker,
                 replay_buffer: PreferenceReplayBuffer,
                 setup_dataloader_func: Callable,
                 logger: Logger,
                 ckpt_saving_func: Callable,
                 beta: float = 0.5,
                 experience_batch_size: int = 512,
                 edpo_preference_batch_size: int = 2,
                 max_epochs: int = 1,
                 tokenizer: Optional[Callable[[Any], dict]] = None,
                 sample_replay_buffer: bool = False,
                 clip_grad: bool = False,
                 callbacks: List[Callback] = [],
                 **generate_kwargs) -> None:
        super().__init__(experience_maker=experience_maker,
                    replay_buffer=replay_buffer,
                    experience_batch_size=experience_batch_size,
                    setup_dataloader_func=setup_dataloader_func,
                    max_epochs=max_epochs,
                    tokenizer=tokenizer,
                    sample_replay_buffer=sample_replay_buffer,
                    callbacks=callbacks,
                    logger=logger,
                    **generate_kwargs)
        self.actor = actor
        self.actor_lr_scheduler = actor_lr_scheduler
        self.actor_loss_fn = PreferenceLoss(beta)
        self.actor_optim = actor_optim
        self.ckpt_saving_func = ckpt_saving_func
        self.clip_grad = clip_grad
        self.edpo_preference_batch_size = edpo_preference_batch_size
        self.device = torch.cuda.current_device()


    def training_step(self, batch_dict: Dict[str, torch.Tensor], has_ref_constraint=True) -> Dict[str, float]:
        self.actor.train()

        tic = time.time()

        attention_mask = batch_dict["attention_mask"].to(self.device)
        num_actions = batch_dict["action_mask"].shape[1]
        action_mask = batch_dict["action_mask"].to(self.device)
        sequences = batch_dict["preference_sequences"].to(self.device)
        
        sequences_2d = sequences.view(-1, sequences.shape[-1])
        attention_mask_2d = attention_mask.view(-1, attention_mask.shape[-1]) # (bs*preference_bs, seq_len)
        action_log_probs_flatten = []
        for i in range(0, sequences_2d.shape[0], self.edpo_preference_batch_size):
            sequences_batch = sequences_2d[i:i+self.edpo_preference_batch_size]
            attention_mask_batch = attention_mask_2d[i:i+self.edpo_preference_batch_size]
            action_log_probs_batch = self.actor(sequences_batch, num_actions, attention_mask_batch)   # (bs, seq_len - 1)

    
            action_log_probs_flatten.append(action_log_probs_batch)
        
        action_log_probs = torch.cat(action_log_probs_flatten, dim=0).view(*action_mask.shape)
  
        preference_mask = batch_dict['preference_mask'].to(self.device)
        ref_action_log_probs = batch_dict['ref_action_log_probs'].to(self.device) if has_ref_constraint else None
        logging_rank_0(f"TIMER-training_step preparing data: {time.time() - tic} s elapsed", level='debug')
        tic = time.time()

        actor_loss, reward,_ = self.actor_loss_fn(action_log_probs,
                                        ref_action_log_probs,
                                        action_mask=action_mask,
                                        preference_mask=preference_mask)
        logging_rank_0(f"TIMER-training_step forward: {time.time() - tic} s elapsed", level='debug')
        tic = time.time()


        self.actor.backward(actor_loss)
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)

        self.actor.step()
        self.actor.zero_grad()

        logging_rank_0(f"TIMER-training_step backward: {time.time() - tic} s elapsed", level='debug')
        tic = time.time()


        approx_kl = compute_approx_kl(
            action_log_probs, batch_dict['ref_action_log_probs'].to(self.device), action_mask=action_mask, return_mean=True, dim=-1
        )
        approx_kl = masked_mean(approx_kl, preference_mask).mean().detach().item()

        rsp = {'actor_loss': actor_loss.detach().item(), 'kl_step':approx_kl, 'reward':reward.mean().item()}
        self.logger.log_metrics(rsp, step=self.global_steps, metrics_group='train')
        self.logger.log_metrics({"actor_lr": self.actor_lr_scheduler.get_last_lr()[0]}, step=self.global_steps, metrics_group='lr')

        logging_rank_0(f"TIMER-training_step logging: {time.time() - tic} s elapsed", level='debug')

        torch.cuda.empty_cache()
        gc.collect()
        
        return rsp
    
    @torch.no_grad()
    def validation_step(self, batch_dict: Dict[str, torch.Tensor], has_ref_constraint: bool=True) -> Dict[str, torch.Tensor]:
        self.actor.eval()

        attention_mask = batch_dict["attention_mask"].to(self.device)
        num_actions = batch_dict["action_mask"].shape[1]
        action_mask = batch_dict["action_mask"].to(self.device)
        sequences = batch_dict["preference_sequences"].to(self.device) #(bs, preference_bs, seq_len)
        
        sequences_2d = sequences.view(-1, sequences.shape[-1])
        attention_mask_2d = attention_mask.view(-1, attention_mask.shape[-1]) # (bs*preference_bs, seq_len)

        action_log_probs_flatten = []

        for i in range(0, sequences_2d.shape[0], self.edpo_preference_batch_size):
            sequences_batch = sequences_2d[i:i+self.edpo_preference_batch_size]
            attention_mask_batch = attention_mask_2d[i:i+self.edpo_preference_batch_size]
            action_log_probs_batch = self.actor(sequences_batch, num_actions, attention_mask_batch)   # (bs, seq_len - 1)
            action_log_probs_flatten.append(action_log_probs_batch)
        
        action_log_probs = torch.cat(action_log_probs_flatten, dim=0).view(*action_mask.shape)        

        preference_mask = batch_dict['preference_mask'].to(self.device)
        ref_action_log_probs = batch_dict['ref_action_log_probs'].to(self.device) if has_ref_constraint else None

        preference_loss, reward, loss_sample = self.actor_loss_fn(action_log_probs,
                                        ref_action_log_probs,
                                        action_mask=action_mask,
                                        preference_mask=preference_mask)
        
        approx_kl = compute_approx_kl(
            action_log_probs, batch_dict['ref_action_log_probs'].to(self.device), action_mask=action_mask, return_mean=True, dim=-1
        )
        approx_kl = masked_mean(approx_kl, preference_mask).mean().detach().item()


        return {'loss':preference_loss.detach().item(), 'kl': approx_kl, 'reward': reward, 'loss_sample': loss_sample}

    def fit(self,
            inputs,
            val_inputs:Optional[Dict]=None,
            seed: int = 42,
            num_episodes: int = 50000,
            max_timesteps: int = 500,
            update_timesteps: int = 5000,
            ignore_ref_first_n_steps: int = 10,
            val_check_interval:int=2,
            save_every_n_episode:int=1) -> None:
        """_summary_

        Args:
            inputs (List[Dict]): training input
            val_inputs (List[Dict], optiional): validating input, defaults to None.
            num_episodes (int, optional): _description_. Defaults to 50000.
            max_timesteps (int, optional): _description_. Defaults to 500.
            update_timesteps (int, optional): _description_. Defaults to 5000.
            val_check_interval (float, optional): 验证频率，每过几个episode验证. Defaults to 2.

        Returns:
            _type_: _description_
        """

        
        # 启动验证集的pipeline
        time = 0
        self._on_fit_start()
        
        if save_every_n_episode == val_check_interval and save_every_n_episode > 1:
            save_every_n_episode -= 1 
        # episode循环
        for episode in range(num_episodes):
            # 验证
            if episode % val_check_interval == 0 and val_inputs is not None:
                self._on_make_experience_start()
                experiences = self._make_experience(val_inputs, is_val=True)
                self.replay_buffer.append(experiences)

                val_dataloader = self.setup_dataloader_func(self.replay_buffer, episode, 0, val_str="_val_")

                sum_info = defaultdict(list)
                for batch in val_dataloader:
                    dicts = self.validation_step(batch, has_ref_constraint=(episode > ignore_ref_first_n_steps))
                    for key,value in dicts.items():
                        sum_info[key].append(value)
                    sum_info['task'].extend(batch['task'])

                self._log_val_metrics(sum_info)
                self.replay_buffer.clear()
                continue
            
            self.timers('episode').start()
            self._on_episode_start(episode)
            # 采样训练
            for timestep in tqdm(range(max_timesteps),
                                desc=f'Episode [{episode+1}/{num_episodes}]',
                                disable=not is_rank_0()):
                time += 1
                sampled_inputs = self._sample_prompts(inputs, seed+timestep+episode)    # 采样训练的prompt

                self.timers('make_experience').start()
                self._on_make_experience_start()
                self.make_exp_steps += 1
                experiences = self._make_experience(sampled_inputs)
                self._on_make_experience_end(experiences)
                self.replay_buffer.append(experiences)
                self.timers('make_experience').stop()
                logging_rank_0(f"TIMER-make_experience: {self.timers('make_experience').elapsed_} s elapsed", level='debug')

                
                # 启动训练
                if time % update_timesteps == 0:
                    make_exp_time = self.timers('make_experience').elapsed(reset=True)
                    self.logger.log_metrics({'make_exp_samples_per_second': len(self.replay_buffer)/make_exp_time},
                                            step=self.global_steps, metrics_group='timers')

                    self._learn(episode, timestep, has_ref_constraint=(episode > ignore_ref_first_n_steps))
                    self.replay_buffer.clear()
                
                torch.cuda.empty_cache()
                gc.collect()
                   

            self._on_episode_end(episode, save_every_n_episode)
            self.timers('episode').stop()
            self.timers.write(['episode'], self.global_steps, reset=True, metrics_group='timers')
        self._on_fit_end()

    def _learn(self, episode, timestep, **kwargs):
        # replay buffer may be empty at first, we should rebuild at each training
        self.timers('learn').start()
        samples = 0
        if self.sample_replay_buffer:
            pbar = tqdm(range(self.max_epochs), desc='Train epoch', disable=not is_rank_0())
            for _ in pbar:
                experience = self.replay_buffer.sample()
                self.train_steps += 1
                metrics = self.training_step(experience)
                pbar.set_postfix(metrics)
                samples += experience['preference_sequences'].shape[0]
        else:
            dataloader = self.setup_dataloader_func(self.replay_buffer, episode, timestep)
            for epoch in range(self.max_epochs):
                self._on_learn_epoch_start(epoch)
                if hasattr(dataloader.sampler, "set_epoch"):
                    dataloader.sampler.set_epoch(epoch)
                pbar = tqdm(
                    dataloader, desc=f'Train epoch [{epoch+1}/{self.max_epochs}]', disable=not is_rank_0())
                for experience in pbar:
                    self._on_learn_batch_start()
                    self.train_steps += 1
                    metrics = self.training_step(experience, **kwargs)
                    self._on_learn_batch_end(metrics, experience)
                    pbar.set_postfix(metrics)
                    samples += experience['preference_sequences'].shape[0]
                    torch.cuda.empty_cache()
                    gc.collect()
                self._on_learn_epoch_end(epoch)
        
        self.timers('learn').stop()
        learn_time = self.timers('learn').elapsed(reset=True)
        logging_rank_0(f"TIMER-learn: {learn_time} s elapsed", level='debug')

        self.logger.log_metrics({'learn_samples_per_second': samples/learn_time},
                                step=self.global_steps, metrics_group='timers')
        self.logger.log_metrics({'episode_step': episode}, step=self.global_steps, metrics_group='train')

    
    def _make_experience(self, inputs: List[Dict], is_val:bool=False) -> List[PreferenceExperience]:
        """采样经验池

        Args:
            inputs (List[Dict]): [{'task':str, 'preference_sequences':torch.Tensor, 'input_ids':torch.Tensor}, ...]
            is_val (bool, optional): 是否是验证集. Defaults to False.

        Returns:
            List[Experience]: _description_
        """
        
        # 调用experience maker，采样经验池
        exp = self.experience_maker.make_experience(inputs, return_kl=False, do_search=(not is_val))

        return exp
    
    
    def _on_make_experience_start(self) -> None:
        # 在开始生成exp的时候，把grad全部设为None，不然grad依旧占用显存
        self.actor.zero_grad()
        return super()._on_make_experience_start()

    def _on_make_experience_end(self, experiences: List[PreferenceExperience]) -> None:
        return super()._on_make_experience_end(experiences)

    def _on_learn_epoch_start(self, epoch: int) -> None:
        return super()._on_learn_epoch_start(epoch)

    def _on_learn_epoch_end(self, epoch: int) -> None:
        return super()._on_learn_epoch_end(epoch)

    def _on_learn_batch_start(self) -> None:
        return super()._on_learn_batch_start()

    def _on_learn_batch_end(self, metrics: dict, experience: PreferenceExperience) -> None:
        return super()._on_learn_batch_end(metrics, experience)
    
    def _on_episode_start(self, episode: int) -> None:
        self.actor_lr_scheduler.step()
        return super()._on_episode_start(episode)

    def _on_episode_end(self, episode: int, save_every_n_episode: int) -> None:
        # saving actor
        if episode % save_every_n_episode == 0:
            self.ckpt_saving_func(episode, self.actor)
        return super()._on_episode_end(episode)
    
    def _log_val_metrics(self, sum_info: Dict[str, List]) -> None:
        """
        Args:
            val_loss (Tensor): _description_
        """
        rank = mpu.get_data_parallel_rank()
        world_size = mpu.get_data_parallel_world_size()
        group = mpu.get_data_parallel_group()

        local_tasks = list(set(sum_info['task']))
        local_tasks_bytes = ','.join(local_tasks).encode() # convert task lists to string

        # encode task
        local_tasks_tensor = torch.tensor([c for c in local_tasks_bytes]).long().to(self.device) #  convert string to tensor
        local_tasks_size = torch.tensor([local_tasks_tensor.numel()]).long().to(self.device)
        gathered_sizes = [torch.zeros(1, dtype=torch.long).to(self.device) for _ in range(world_size)]
        dist.all_gather(gathered_sizes, local_tasks_size, group=group)

        # merge task
        max_size = max([size.item() for size in gathered_sizes])
        gathered_tasks = [torch.zeros(max_size, dtype=torch.long).to(self.device) for _ in range(world_size)]
        gathered_tasks[rank][:local_tasks_tensor.numel()] = local_tasks_tensor
        dist.all_gather(gathered_tasks, gathered_tasks[rank], group=group)

        #decode task list
        global_tasks = []
        for tasks in gathered_tasks:
            tasks_bytes = ''.join(chr(c) for c in tasks.tolist() if c > 0)
            global_tasks.extend(tasks_bytes.split(','))
        
        global_tasks = list(set(global_tasks))

        local_sums = {task:torch.tensor([0., 0., 0.]).to(self.device) for task in global_tasks} 
        # (reward_sum, loss_sum, count)
        rewards = [element.item() for reward in sum_info['reward'] for element in reward]
        losses = [element.item() for loss in sum_info['loss_sample'] for element in loss]
        for task in global_tasks:
            task_reward = [r for t,r in zip(sum_info['task'], rewards) if t == task ]
            task_loss = [l for t,l in zip(sum_info['task'], losses) if t == task ]
            local_sums[task][0] += sum(task_reward) if len(task_reward) > 0 else 0
            local_sums[task][1] += sum(task_loss) if len(task_loss) > 0 else 0
            assert len(task_reward) == len(task_loss)

            local_sums[task][2] += len(task_reward) if len(task_reward) > 0 else 0
            dist.all_reduce(local_sums[task], op=dist.ReduceOp.SUM, group=group)

            reward_sum, loss_sum, count = local_sums[task]
            self.logger.log_metrics({task: reward_sum/count}, step=self.global_steps, metrics_group='val_reward')
            self.logger.log_metrics({task: loss_sum/count}, step=self.global_steps, metrics_group='val_loss')

        global_reward_sum, global_loss_sum, global_count = [sum(x) for x in zip(*list(local_sums.values()))]
        global_reward_mean = global_reward_sum/global_count
        global_loss_mean = global_loss_sum/global_count
        self.logger.log_metrics({'all': global_reward_mean}, step=self.global_steps, metrics_group='val_reward')
        self.logger.log_metrics({'all': global_loss_mean}, step=self.global_steps, metrics_group='val_loss')

       
        # mean
        sum_keys = ['kl']
        for key,item in sum_info.items():
            if key not in sum_keys:
                continue
            item_sum = torch.Tensor(item).to(self.device)
            global_item_mean,_,_ = get_global_statistics(item_sum)
            self.logger.log_metrics({key: global_item_mean.item()}, step=self.global_steps, metrics_group='val_info')

        return
