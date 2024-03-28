# encoding=utf-8
from dataclasses import dataclass
from typing import Optional

from .base import BasicConfig, BasicMegatronDeepspeedConfig
from .data import DataConfig


@dataclass
class PolicyModelConfig(BasicConfig):
    """ 生成模型配置"""

    # policy model
    policy_model_path:     Optional[str] = None
    policy_tokenizer_path: Optional[str] = None
    policy_max_seq_len:    int = 2048

    def update(self, args: dict):
        return BasicConfig.update(self, args, "policy_model")
    


@dataclass
class RewardModelConfig(BasicConfig):

    # reward model
    reward_model_path:     Optional[str] = None
    rm_tokenizer_path:  Optional[str] = None
    rm_max_seq_len:    int = 2048
    granularity:    str = "token"

    def update(self, args: dict):
        return BasicConfig.update(self, args, "reward_model")
    
@dataclass
class ModelConvertConfig(BasicMegatronDeepspeedConfig, PolicyModelConfig, RewardModelConfig, DataConfig):
    def update(self, args:dict):
        BasicMegatronDeepspeedConfig.update(self, args)
        DataConfig.update(self, args)
        RewardModelConfig.update(self, args)
        PolicyModelConfig.update(self, args)
        return

@dataclass
class ReflectModelConfig(BasicConfig):

    # reflect model
    reflector_model_path:           Optional[str] = None
    reflector_tokenizer_path:       Optional[str] = None
    reflector_max_seq_len:          int = 2048

    def update(self, args: dict):
        return BasicConfig.update(self, args, "reflect_model")