import gc
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import Optimizer
from fengshen_inner.models.megatron import mpu
from chatgpt.experience_maker import (SandBoxExperience, SandBoxExperienceMaker,
                                      SandBoxExperienceV2, SandBoxExperienceMakerV2)
from chatgpt.replay_buffer import FDReplayBuffer
from chatgpt.nn import Actor, GeneralizedKDLoss, EOSLoss
from chatgpt.nn.utils import shrink_mask, zero_pad_sequences
from chatgpt.logger.base import Logger
from chatgpt.utils import is_rank_0, logging_rank_0, CostTimer
from .base import Trainer
from .callbacks import Callback


START_BACKWARD = 1
SKIP_BACKWARD = 0


class FDSandboxTrainer(Trainer):

    def __init__(self,
        actor: Actor,
        actor_optim: Optimizer,
        actor_lr_scheduler,
        actor_tokenizer,
        experience_maker: SandBoxExperienceMaker,
        replay_buffer: FDReplayBuffer,
        setup_dataloader_func: Callable,
        divergence: str = 'JSD',
        JSD_coef: float = 0.5,
        GKD_coef: float = 1.,
        temperature: float = 1.,
        shrink: Tuple[int] = (0, 0),
        level: str = 'token',
        experience_batch_size: int = 512,
        max_epochs: int = 1,
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
        self.JSD_coef = JSD_coef                                        
        self.GKD_coef = GKD_coef           
        self.shrink = shrink      
        self.level = level                       
        self.actor_loss = GeneralizedKDLoss(
            divergence=self.divergence,
            JSD_coef=self.JSD_coef,
            temperature=self.temperature,
            level=self.level
        )

        for key in kwargs:
            logging_rank_0(f"NoImplementation Warning: Attribute '{key}' is not yet supported in 'FDTrainer'.")     

        ### Other ###
        self.ckpt_saving_func = ckpt_saving_func
        self.exp_saving_func = exp_saving_func
        self.sync_info = torch.tensor([0], dtype=torch.int, device=self.actor.device)
        self.training_steps = 0
        self.experience_steps = 0
        self.backward_steps = 0


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

    def training_step(self, experience: SandBoxExperienceV2):
        
        ### prepare
        batchsize = experience.batchsize            
        self.actor.train()
        num_actions = None
        self.training_steps += 1
        self.logger.log_metrics({"actor_lr": self.actor_lr_scheduler.get_last_lr()[0]}, step=self.logging_steps, metrics_group='training')

        ### forward and loss
        log_prob, logits = self.actor(
            experience.sequences, num_actions, experience.attention_mask, return_logits=True)
        with torch.no_grad():
            ref_log_prob, ref_logits = self.actor(
                experience.ref_sequences, num_actions, experience.ref_attention_mask, return_logits=True)
        GKD_loss = self.actor_loss(
            logits, ref_logits, shrink_mask(experience.action_mask, self.shrink), shrink_mask(experience.ref_action_mask, self.shrink))

        ### backward
        self._on_actor_backward_start()
        if self.do_actor_backward:
            self.backward_steps += 1
            with CostTimer():
                self.actor.backward(GKD_loss)
                self.actor.step()
                self.actor.zero_grad()
            self.logger.log_metrics({"backward_time": CostTimer.get_time()}, step=self.logging_steps, metrics_group='training')
        self.logger.log_metrics({"backward_steps": self.backward_steps}, step=self.logging_steps, metrics_group='training')
        gc.collect()
        torch.cuda.empty_cache()

        ### logging
        action_log_prob = (log_prob * experience.action_mask).sum() / batchsize
        ref_action_log_prob = (ref_log_prob * experience.ref_action_mask).sum() / batchsize
        rsp = {'GKD_loss': GKD_loss.item(),
               'action_log_prob': action_log_prob.item(),
               'ref_action_log_prob': ref_action_log_prob.item()}
        self.logger.log_metrics(rsp, step=self.logging_steps, metrics_group='loss')  

        ### test
        self.actor.eval()
        with torch.no_grad():
            for i in range(batchsize):
                self.experience_steps += 1
                log_prob = self.actor(experience.test_sequences[i], num_actions, experience.test_attention_mask[i])
                option_prob = (log_prob * experience.test_action_mask[i]).sum(dim=-1)  
                options = {}
                options['test_ans'] = experience.test_labels[i].item()
                options['test_pred'] = option_prob.argmax().item()

                probs = {}
                probs['ans_prob'] = option_prob[options['test_ans']].item()
                probs['idk_prob'] = option_prob[-1].item()
                probs['max_prob'] = option_prob[options['test_pred']].item()
                probs['err_prob'] = torch.tensor([option_prob[idx].item() for idx in range(len(option_prob) - 1) if idx != experience.test_labels[i]]).mean().item()

                length = experience.attention_mask[i].sum().item()
                self.logger.log_metrics({"length": length}, step=self.logging_steps, metrics_group='experience')
                self.logger.log_metrics(probs, step=self.logging_steps, metrics_group='probs')
                self.logger.log_metrics(options, step=self.logging_steps, metrics_group='options')

                exp_dict = {
                    'ref_text': self.tokenizer.decode(experience.ref_sequences[i], skip_special_tokens=True),
                    'length': length,
                    'loss': GKD_loss.item(),
                    'ori_log_prob': action_log_prob.item(),
                    'ref_log_prob': ref_action_log_prob.item(),
                }
                # js = self.top_possible_tokens(experience.sequences, logits, experience.action_mask,
                #                               ref_logits, experience.ref_action_mask)
                self.exp_saving_func(exp_dict, None, step=self.logging_steps)

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
                self.actor.eval()
                if val_saving_func is None:
                    logging_rank_0("Validation is not implemented yet.", level='info')


    @property
    def logging_steps(self):
        return self.training_steps + self.experience_steps

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

        sampled_indices = torch.randperm(len(prompts), generator=g).tolist()[:self.experience_batch_size]
        sampled_prompts = [prompts[i] for i in sampled_indices]
        
        return sampled_prompts

    def _make_experience(self, prompts: List[str]) -> SandBoxExperience:
        experience = self.experience_maker.make_experience(inputs=prompts)
        return experience
    
    def top_possible_tokens(self, seq, logits, mask, ref_logits, ref_mask):
        if not is_rank_0():
            return
        
        js = {}
        action = torch.masked_select(seq, F.pad(mask, (1, 0), value=False)).tolist()
        logits, mask = self.actor_loss._dim_check(logits, mask)
        ref_logits, ref_mask = self.actor_loss._dim_check(ref_logits, ref_mask)
        prob = self.actor_loss._build_probs(logits, mask, False)
        ref_prob = self.actor_loss._build_probs(ref_logits, ref_mask, False)
        
        value, idx = torch.topk(prob, k=10, dim=-1)
        ref_value, ref_idx = torch.topk(ref_prob, k=10, dim=-1)

        js['length'] = len(action)
        for i, act in enumerate(action):
            pos_i = {}
            pos_i['token'] = act
            pos_i['prob'] = prob[i, act].item()
            pos_i['ref_prob'] = ref_prob[i, act].item()
            pos_i['topk'] = {
                'values': value[i].tolist(),
                'indices': idx[i].tolist(),
                'ref_values': ref_value[i].tolist(),
                'ref_indices': ref_idx[i].tolist()
            }
            js[i] = pos_i
        js = json.dumps(js)
        return js