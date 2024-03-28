# encoding=utf-8
import os
import socket
from datetime import datetime
from typing import Any, Dict, Optional

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from chatgpt.utils import logging_rank_0

from ..utils import local_rank
from .base import Logger


class WandbLogger(Logger):
    def __init__(self,
                 project: str = None,
                 group: str = None,
                 entity: str = None,
                 ignore: bool = False,
                 name: str = None,
                 tensorboard_dir: str = None):
        """
        ignore: 忽略这个logger， 不起作用，使用情况比如说一个机器，只有local_rank或者global_rank=0才需要输出之类的
        """
        self.ignore = ignore
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if ignore == True:
            return
        if name == None:
            self.name = f"{socket.gethostname()}-{local_rank()}-{curr_time}"
        else:
            self.name = name
        wandb.init(
            project=project,
            group=group,
            name=self.name,
            entity=entity,
            save_code=False,
            force=False,
        )
        
        self.tb_writer = None
        if tensorboard_dir is not None:
            self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)
            logging_rank_0(f"Enable Tensorboard Logger, save to '{tensorboard_dir}'.")
            
        return

    @property
    def log_dir(self) -> Optional[str]:
        return None

    def log_hyperparams(self, params: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        if self.ignore:
            return
        wandb.config.update(params)

    def log_metrics(self, 
                    metrics: Dict[str, float], 
                    step: Optional[int] = None, 
                    only_rank0: Optional[bool] = False, 
                    metrics_group: Optional[str] = None,
                    *args: Any, **kwargs: Any) -> None:
        if self.ignore:
            return
        if only_rank0 and torch.distributed.get_rank() != 0:
            return
        if metrics_group is not None:
            for k, v in metrics.items():
                wandb.log({"/".join([metrics_group, k]) : v}, step=step)
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar(
                        tag="/".join([metrics_group, k]),
                        scalar_value=v,
                        global_step=step
                    )
        else:
            wandb.log(metrics, step=step)
            if self.tb_writer is not None:
                for k, v in metrics.items():
                    self.tb_writer.add_scalar(
                        tag=k,
                        scalar_value=v,
                        global_step=step,
                    )
        return
