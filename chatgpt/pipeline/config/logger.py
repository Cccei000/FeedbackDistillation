# encoding=utf-8
from dataclasses import dataclass
from typing import Optional

from .base import BasicConfig


@dataclass
class LoggerConfig(BasicConfig):
    """ logger 配置 """
    
    wandb_project:  str             = "Fengshen-HumanFeedback"
    wandb_group:    str             = ""
    wandb_name:     str             = ""
    wandb_team:     Optional[str]   = None
    
    def update(self, args: dict):
        return BasicConfig.update(self, args, "logger")

