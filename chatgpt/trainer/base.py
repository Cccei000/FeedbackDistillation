# encoding=utf-8
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Union, Optional
from chatgpt.logger.base import Logger
import torch
from chatgpt.experience_maker import Experience, ExperienceMaker
from chatgpt.replay_buffer import ReplayBuffer
from torch import Tensor
from tqdm import tqdm
import numpy as np

from .callbacks import Callback
from ..utils import is_rank_0, Timers


class FixedKLController:
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass

class AdaptiveKLController:
    """Adaptive KL Controller as described in Ziegler et al. "Fine-Tuning Language Models from Human Preferences"
    Reference: Section 2.2 https://arxiv.org/pdf/1909.08593.pdf#page=2
    Source: https://github.com/openai/lm-human-preferences/blob/master/lm_human_preferences/train_policy.py
    """

    def __init__(self, kl_coef: float, target: float, horizon: int):
        self.kl_coef = kl_coef
        self.value = kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int=1):
        """Returns adaptively updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)  # ϵₜ
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult  # βₜ₊₁
        self.value = min(max(self.value, 1e-19), self.kl_coef) ## 裁剪

        return

class Trainer(ABC):
    """
        Base class for rlhf trainers.

    Args:
        experience_maker (ExperienceMaker): the experience maker to use for produce experience to fullfill replay buffer
        replay_buffer (ReplayBuffer): the replay buffer to use for training
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        tokenizer (Callable, optional): the tokenizer to use for tokenizing the input
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
        setup_dataloader_func: use to setup dataloader for sample_replay_buffer
    """

    def __init__(self,
                 experience_maker: ExperienceMaker,
                 replay_buffer: ReplayBuffer,
                 setup_dataloader_func: Callable,
                 logger: Logger,
                 experience_batch_size: int = 512,
                 max_epochs: int = 1,
                 tokenizer: Optional[Callable[[Any], dict]] = None,
                 sample_replay_buffer: bool = False,
                 callbacks: List[Callback] = [],
                 **generate_kwargs) -> None:
        super().__init__()
        self.experience_maker = experience_maker
        self.replay_buffer = replay_buffer
        self.experience_batch_size = experience_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.sample_replay_buffer = sample_replay_buffer
        self.callbacks = callbacks
        self.setup_dataloader_func = setup_dataloader_func
        self.logger = logger
        self.timers = Timers(logger)

        # 用来做监控的一些参数
        self.train_steps = 0
        self.make_exp_steps = 0

    @abstractmethod
    def training_step(self, experience: Experience) -> Dict[str, Any]:
        pass
    
    def _make_experience(self, inputs: Union[Tensor, Dict[str, Tensor]]) -> List[Experience]:
        if isinstance(inputs, Tensor):
            return self.experience_maker.make_experience(inputs, **self.generate_kwargs)
        elif isinstance(inputs, dict):
            return self.experience_maker.make_experience(**inputs, **self.generate_kwargs)
        else:
            raise ValueError(f'Unsupported input type "{type(inputs)}"')
    
    def _sample_prompts(self, prompts:Union[Dict[str,List[str]],List[str]], seed:int) -> List[str]:
        """
            随机选取用于生成的Prompt，如果存在任务标签，则按照任务类型进行分层采样

        Args:
            prompts (Union[Dict[str,List[str]],List[str]]): _description_
            seed (int): _description_

        Returns:
            List[str]: _description_
        """
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        if isinstance(prompts, list):
            sampled_indices = torch.randperm(len(prompts), generator=g).tolist()[:self.experience_batch_size]
            return [prompts[i] for i in sampled_indices]
        
        # 计算每个task需要采样的prompt数量
        ## 任务总数
        num_free_task = len(prompts)
        ## 每个数据集的样本数量
        num_data_per_free_task = int(self.experience_batch_size / num_free_task) if num_free_task > 0 else 0
        
        output_free_prompts = []
            
        # 分层采样
        for t_name, queries in prompts.items():
            sampled_indices = torch.randperm(len(queries), generator=g).tolist()[:num_data_per_free_task]
            selected_free_prompts = [queries[i] for i in sampled_indices]
            output_free_prompts.extend(selected_free_prompts)
        
        return output_free_prompts

    def _learn(self, episode, timestep):
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
                samples += experience.sequences.shape[0]
        else:
            dataloader = self.setup_dataloader_func(self.replay_buffer, episode, timestep)
            device = torch.cuda.current_device()
            for epoch in range(self.max_epochs):
                self._on_learn_epoch_start(epoch)
                if hasattr(dataloader.sampler, "set_epoch"):
                    dataloader.sampler.set_epoch(epoch)
                pbar = tqdm(
                    dataloader, desc=f'Train epoch [{epoch+1}/{self.max_epochs}]', disable=not is_rank_0())
                for experience in pbar:
                    self._on_learn_batch_start()
                    experience.to_device(device)
                    self.train_steps += 1
                    metrics = self.training_step(experience)
                    self._on_learn_batch_end(metrics, experience)
                    # pbar.set_postfix(metrics)
                    samples += experience.sequences.shape[0]
                self._on_learn_epoch_end(epoch)
        self.timers('learn').stop()
        learn_time = self.timers('learn').elapsed(reset=True)
        self.logger.log_metrics({'learn_samples_per_second': samples/learn_time},
                                step=self.global_steps, metrics_group='timers')

    def fit(self, prompts, num_episodes: int = 50000, max_timesteps: int = 500, update_timesteps: int = 5000) -> None:
        time = 0
        self._on_fit_start()
        for episode in range(num_episodes):
            self.timers('episode').start()
            self._on_episode_start(episode)
            for timestep in tqdm(range(max_timesteps),
                                 desc=f'Episode [{episode+1}/{num_episodes}]',
                                 disable=not is_rank_0()):
                time += 1
                rand_prompts = self._sample_prompts(prompts)
                if self.tokenizer is not None:
                    inputs = self.tokenizer.batch_encode_plus(rand_prompts)["input_ids"]
                else:
                    inputs = rand_prompts

                self.timers('make_experience').start()
                self._on_make_experience_start()
                self.make_exp_steps += 1
                experiences = self._make_experience(inputs)
                self._on_make_experience_end(experiences)
                self.replay_buffer.append(experiences)
                self.timers('make_experience').stop()

                if time % update_timesteps == 0:
                    make_exp_time = self.timers('make_experience').elapsed(reset=True)
                    self.logger.log_metrics({'make_exp_samples_per_second': len(self.replay_buffer)/make_exp_time},
                                            step=self.global_steps, metrics_group='timers')

                    self._learn(episode, timestep)
                    self.replay_buffer.clear()

            self._on_episode_end(episode)
            self.timers('episode').stop()
            self.timers.write(['episode'], self.global_steps, reset=True, metrics_group='timers')
        self._on_fit_end()

    def _on_fit_start(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_start()

    def _on_fit_end(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_end()

    def _on_episode_start(self, episode: int) -> None:
        for callback in self.callbacks:
            callback.on_episode_start(episode)

    def _on_episode_end(self, episode: int) -> None:
        for callback in self.callbacks:
            callback.on_episode_end(episode)

    def _on_make_experience_start(self) -> None:
        for callback in self.callbacks:
            callback.on_make_experience_start()

    def _on_make_experience_end(self, experiences: List[Experience]) -> None:
        for callback in self.callbacks:
            callback.on_make_experience_end(experiences)

    def _on_learn_epoch_start(self, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_learn_epoch_start(epoch)

    def _on_learn_epoch_end(self, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_learn_epoch_end(epoch)

    def _on_learn_batch_start(self) -> None:
        for callback in self.callbacks:
            callback.on_learn_batch_start()

    def _on_learn_batch_end(self, metrics: dict, experience: Experience) -> None:
        for callback in self.callbacks:
            callback.on_learn_batch_end(metrics, experience)

    @property
    def global_steps(self) -> int:
        return self.train_steps + self.make_exp_steps
