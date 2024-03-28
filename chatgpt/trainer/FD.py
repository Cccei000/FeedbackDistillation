import gc
from typing import Any, Callable, Dict, List, Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import Optimizer
from fengshen_inner.models.megatron import mpu
from chatgpt.experience_maker import FDExperience, FDExperienceMaker
from chatgpt.replay_buffer import FDReplayBuffer
from chatgpt.nn import Actor, GeneralizedKDLoss, EOSLoss
from chatgpt.nn.utils import shrink_mask
from chatgpt.logger.base import Logger
from chatgpt.utils import is_rank_0, logging_rank_0, CostTimer
from .base import Trainer
from .callbacks import Callback

START_BACKWARD = 1
SKIP_BACKWARD = 0


class FDTrainer(Trainer):

    def __init__(self,
        actor: Actor,
        actor_optim: Optimizer,
        actor_lr_scheduler,
        actor_tokenizer,
        experience_maker: FDExperienceMaker,
        replay_buffer: FDReplayBuffer,
        setup_dataloader_func: Callable,
        divergence: str = 'JSD',
        JSD_coef: float = 0.5,
        GKD_coef: float = 1.,
        temperature: float = 1.,
        shrink: Tuple[int] = (0, 0),
        experience_batch_size: int = 512,
        max_epochs: int = 1,
        skip_exp: bool = False,
        sample_replay_buffer: bool = False,
        ckpt_saving_func: Optional[Any] = None,
        exp_saving_func: Optional[Any] = None,
        logger: Optional[Logger] = None,
        callbacks: List[Callback] = [],
        **kwargs) -> None:

        super().__init__(
            experience_maker=experience_maker,
            replay_buffer=replay_buffer,
            experience_batch_size=experience_batch_size,
            setup_dataloader_func=setup_dataloader_func,
            max_epochs=max_epochs,
            tokenizer=actor_tokenizer,
            sample_replay_buffer=sample_replay_buffer,
            callbacks=callbacks,
            logger=logger,
        )
        
        ### Model ###
        self.actor = actor
        self.actor_optim = actor_optim
        self.actor_lr_scheduler = actor_lr_scheduler

        ### Loss ###
        self.divergence = divergence.upper().strip()
        self.temperature = temperature
        self.shrink = shrink
        self.JSD_coef = JSD_coef                                        
        self.GKD_coef = GKD_coef                                        
        self.actor_loss = GeneralizedKDLoss(
            divergence=self.divergence,
            JSD_coef=self.JSD_coef,
            temperature=self.temperature
        )
        self.eos_loss = EOSLoss(self.tokenizer.eos_token_id)

        ### NotImplemented ###
        # self.clip_grad = clip_grad
        # self.constrain_actor_kl_coef = constrain_actor_kl_coef         
        # self.target_constrain_actor_kl = target_constrain_actor_kl     
        # self.kl_adaptor_horizon = kl_adaptor_horizon                   
        # self.constraint_actor = constraint_actor                        
        # self.update_constraint_actor_interval = update_constraint_actor_interval
        # self.mixed_sampling = mixed_sampling                        
        # self.separate_sampling = separate_sampling
        for key in kwargs:
            logging_rank_0(f"NoImplementation Warning: Attribute '{key}' is not yet supported in 'FDTrainer'.")     

        ### Other ###
        self.skip_exp = skip_exp
        self.ckpt_saving_func = ckpt_saving_func
        self.exp_saving_func = exp_saving_func
        self.sync_info = torch.tensor([0], dtype=torch.int, device=self.actor.device)
        self.training_steps = 0
        self.backward_steps = 0
        self.exp_steps = 0


    def _learn(self, episode, timestep):
        device = torch.cuda.current_device()
        if self.sample_replay_buffer:
            pbar = tqdm(range(self.max_epochs), desc='Train epoch', disable=not is_rank_0())
            for epoch in pbar:
                experience = self.replay_buffer.sample()
                experience.to_device(device)
                metrics = self.training_step(experience)
                pbar.set_postfix(metrics)
        else:
            dataloader = self.setup_dataloader_func(self.replay_buffer)
            for epoch in range(self.max_epochs):
                pbar = tqdm(
                    dataloader, desc=f'Train epoch [{epoch+1}/{self.max_epochs}]', disable=not is_rank_0())
                for experience in pbar:
                    experience.to_device(device)
                    metrics = self.training_step(experience)
                    pbar.set_postfix(metrics)        

    def training_step(self, experience: FDExperience):
        
        ### prepare
        batchsize = experience.sequences.shape[0]
        if batchsize == 1 and self.skip_exp:
            if experience.scores.item() == -1 or experience.repeating.item() == 1:
                return {} # skip bad samples
            
        self.actor.train()
        num_actions = None
        self.training_steps += 1
        self.logger.log_metrics({"actor_lr": self.actor_lr_scheduler.get_last_lr()[0]}, step=self.training_steps, metrics_group='training')

        ### forward and loss
        log_prob, logits = self.actor(
            experience.sequences, num_actions, experience.attention_mask, return_logits=True)
        with torch.no_grad():
            ref_log_prob, ref_logits = self.actor(
                experience.ref_sequences, num_actions, experience.ref_attention_mask, return_logits=True)
        GKD_loss = self.actor_loss(
            logits, ref_logits, shrink_mask(experience.action_mask, self.shrink), shrink_mask(experience.ref_action_mask, self.shrink))
        EOS_loss = -self.eos_loss(log_prob, experience.action_mask)
        total_loss = self.GKD_coef * GKD_loss + (1 - self.GKD_coef) * EOS_loss

        ### backward
        self._on_actor_backward_start()
        if self.do_actor_backward:
            self.backward_steps += 1
            with CostTimer():
                self.actor.backward(total_loss)
                self.actor.step()
                self.actor.zero_grad()
            self.logger.log_metrics({"backward_time": CostTimer.get_time()}, step=self.training_steps, metrics_group='training')
        self.logger.log_metrics({"backward_steps": self.backward_steps}, step=self.training_steps, metrics_group='training')
        gc.collect()
        torch.cuda.empty_cache()

        ## logging metrics
        action_log_prob = (log_prob * experience.action_mask).sum() / batchsize
        ref_action_log_prob = (ref_log_prob * experience.ref_action_mask).sum() / batchsize

        for i in range(batchsize):
            self.exp_steps += 1
            self.logger.log_metrics({'exp_length': experience.attention_mask[i].sum().item()}, step=self.exp_steps, metrics_group='exp')
            self.logger.log_metrics({'ref_exp_length': experience.ref_attention_mask[i].sum().item()}, step=self.exp_steps, metrics_group='exp')
            self.logger.log_metrics({'exp_action_length': experience.action_mask[i].sum().item()}, step=self.exp_steps, metrics_group='exp')

        rsp = {'GKD_loss': GKD_loss.item(),
               'EOS_loss': EOS_loss.item(),
               'total_loss': total_loss.item(),
               'action_log_prob': action_log_prob.item(),
               'ref_action_log_prob': ref_action_log_prob.item()}
        self.logger.log_metrics(rsp, step=self.training_steps, metrics_group='loss')       
        if batchsize == 1:
            self.logger.log_metrics({'feedback_score': experience.scores.item()}, step=self.training_steps, metrics_group='loss')
            self.logger.log_metrics({'repeating': experience.repeating.item()}, step=self.training_steps, metrics_group='loss')

        ### logging exp
        if batchsize == 1:
            exp_dict = {
                 'text': self.tokenizer.batch_decode(experience.ref_sequences, skip_special_tokens=True)[0],
                 'score': experience.scores.item(),
                 'loss': GKD_loss.item(),
                 'ori_log_prob': action_log_prob.item(),
                 'ref_log_prob': ref_action_log_prob.item()
            }
            self.exp_saving_func(exp_dict, step=self.training_steps)

        gc.collect()
        torch.cuda.empty_cache()
        return rsp


    def fit(self,
        prompts,
        val_prompts: Optional[Any] = None,
        seed: int = 42,
        num_episodes: int = 50000,
        max_timesteps: int = 500,
        update_timesteps: int = 5000,
        val_check_interval: int = 2,
        val_saving_func: Optional[Callable] = None) -> None:

        for episode in range(num_episodes):
            # self.actor_lr_scheduler.step()
            for timestep in tqdm(range(max_timesteps),
                                 desc=f"Episode [{episode + 1}/{num_episodes}]",
                                 disable=not is_rank_0()):
                rand_prompts = self._sample_prompts(prompts=prompts, seed=seed + episode + timestep)
                self.actor.zero_grad()
                experience = self._make_experience(rand_prompts)
                self.replay_buffer.append(experience)

                if (timestep + 1) % update_timesteps == 0:
                    self._learn(episode, timestep)
                    self.replay_buffer.clear()

                gc.collect()
                torch.cuda.empty_cache()
            
            if (episode + 1) % val_check_interval == 0:
                # self.ckpt_saving_func(episode, self.actor)
                self.actor.eval()
                if val_saving_func is None:
                    logging_rank_0("Validation is not implemented yet.", level='info')


    def _on_actor_backward_start(self) -> None:
        """在Actor反向之前同步不同dp rank的信号，确认是否进行反向

        Args:
            do_backward (bool, optional): 本dp rank是否准备进行反向. Defaults to True.
        """
        
        self.sync_info[0] = START_BACKWARD
        
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
                        
    @property
    def do_actor_backward(self) -> bool:
        """判断是否满足Actor反向传播的条件

        Returns:
            bool: 是否满足Actor反向传播的条件
        """
        return self.sync_info[0] == START_BACKWARD

    def _sample_prompts(self, prompts, seed):

        g = torch.Generator()
        g.manual_seed(seed)

        total_prompts = []

        for task, queries in prompts.items():
            total_prompts.extend(queries)

        sampled_indices = torch.randperm(len(total_prompts), generator=g).tolist()[:self.experience_batch_size]
        sampled_prompts = [total_prompts[i] for i in sampled_indices]
        
        return sampled_prompts

    def _make_experience(self, prompts: List[str]) -> FDExperience:
        experience = self.experience_maker.make_experience(inputs=prompts)
        return experience