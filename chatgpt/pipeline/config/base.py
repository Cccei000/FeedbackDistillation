# encoding=utf-8
import os
from dataclasses import dataclass
from typing import Optional

from chatgpt.utils import logging_rank_0

from .config import AVAILABLE_PP, AVAILABLE_TP

AVAILABLE_PRECISION = ["bf16", "fp16"]


AVAILABLE_LR_SCHEDULER = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau"]


@dataclass
class BasicConfig:
    """ 配置基类 """

    def update(self, args: dict, namespace: Optional[str] = None):

        if namespace:
            config = args.get(namespace, None)
            if config and isinstance(config, dict):
                for key, value in config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                    else:
                        logging_rank_0(f"Illegal Attr: '/{namespace}/{key}'", level="warning")
        return
    
    def check(self):
        return True

    @classmethod
    def check_path(cls, path:str, default_path:Optional[str]=None, model_str:str="") -> str:
        """检查目录是否存在；若不存在返回默认值

        Args:
            path (str): 待检查的路径
            default_path (Optional[str], optional): 默认路径. Defaults to None.
            model_str (str, optional): logger输出. Defaults to "".

        Returns:
            str: 返回路径
        """           
        if not path or not os.path.exists(path):
            level = "warning" if default_path else "error"
            logging_rank_0(f"{model_str}_path ({path}) isn't a directory, set to default." if default_path 
                        else f"{model_str}_path ({path}) isn't a directory.", level)
            path = default_path if default_path else None

        return path


@dataclass
class BasicTrainerConfig(BasicConfig):
    """ 基础 Trainer 配置，对应于 Fengshen_inner 中参数 """

    learning_rate:      float = 1e-6
    min_learning_rate:  float = 1e-7
    lr_decay_steps:     int = 0
    lr_decay_ratio:     float = 1.0
    warmup_steps:       int = 0
    warmup_ratio:       float = 1e-2
    weight_decay:       float = 1e-1
    adam_beta1:         float = 0.9
    adam_beta2:         float = 0.999
    adam_epsilon:       float = 1e-5
    scheduler_type:     str = "cosine"
    warmup_min_lr:      float = 1e-9
    warmup_max_lr:      float = 1e-4

    def update(self, args: dict):
        return BasicConfig.update(self, args, "defaults")
    
    def check(self):
        
        check_res = True
        
        if self.learning_rate <= 0:
            logging_rank_0(msg=f"Value Error: learning_rate ({self.learning_rate}) should be greater than 0.", level="error")
            check_res = False
        
        if self.scheduler_type not in AVAILABLE_LR_SCHEDULER:
            logging_rank_0(msg=f"Value Error: scheduler_type ({self.scheduler_type}) should be in ({AVAILABLE_LR_SCHEDULER}).", level="error")
            check_res = False
        
        return check_res & super().check()


@dataclass
class BasicMegatronDeepspeedConfig(BasicConfig):
    """ Megatron Deepspeed 的配置 """

    tensor_model_parallel_size:     int = 1
    pipe_model_parallel_size:       int = 1
    seed:                           int = 1234
    
    deepspeed_stage:                int = 2
    offload_optimizer:              bool = True
    policy_precision:               str = "bf16"
    rm_precision:                   str = "fp16"
    reflector_precision:            str = "bf16"
    
    world_size:                     int = 64
    train_micro_batch_size_per_gpu: int = 1
    loss_scale:                     float = 0.0
    initial_scale_power:            int = 16
    loss_scale_window:              int = 1000
    hysteresis:                     int = 2
    min_loss_scale:                 int = 2

    lora_rank:                      int = 0

    gradient_accumulation_steps:    int = 2
    
    enable_hybrid_engine:           bool = False
    
    def update(self, args: dict):
        return BasicConfig.update(self, args, "megatron")

    def check(self):
        
        check_res = True
        
        if self.tensor_model_parallel_size not in AVAILABLE_TP:
            logging_rank_0(msg=f"tensor_model_parallel_size ({self.tensor_model_parallel_size}) should in {set(AVAILABLE_TP)}.", level="error")
            check_res = False
        
        if self.pipe_model_parallel_size not in AVAILABLE_PP:
            logging_rank_0(msg=f"pipe_model_parallel_size ({self.pipe_model_parallel_size}) should in {set(AVAILABLE_PP)}.", level="error")
            check_res = False
            
        if self.policy_precision not in AVAILABLE_PRECISION:
            logging_rank_0(msg=f"policy_precision ({self.policy_precision}) should in {set(AVAILABLE_PRECISION)}.", level="error")
            check_res = False
            
        if self.rm_precision not in AVAILABLE_PRECISION:
            logging_rank_0(msg=f"rm_precision ({self.rm_precision}) should in {set(AVAILABLE_PRECISION)}.", level="error")
            check_res = False

        if self.reflector_precision not in AVAILABLE_PRECISION:
            logging_rank_0(msg=f"reflector_precision ({self.reflector_precision}) should in {set(AVAILABLE_PRECISION)}.", level="error")
            check_res = False
        
        if self.lora_rank < 0 or not isinstance(self.lora_rank, int):
            logging_rank_0(msg=f"lora_rank ({self.lora_rank}) should be an integer and not smaller than 0.", level="error")
            check_res = False
        
        return check_res & super().check()


@dataclass
class BasicGenerateConfig(BasicConfig):

    top_p:              float = 0.85
    top_k:              int = 0
    repetition_penalty: float = 1.0
    temperature:        float = 0.85
    max_new_tokens:      int = 1024

    def update(self, args: dict):
        return BasicConfig.update(self, args, "generation")
    
    def check(self):
        
        check_res = True
        
        if self.top_p < 0.0 or self.top_p > 1.0:
            logging_rank_0(msg=f"top_p ({self.top_p}) should be in [0.0, 1.0].", level="error")
            check_res = False 
        
        if self.top_k < 0.0:
            logging_rank_0(msg=f"top_k ({self.top_p}) should be greater than or equal to 0.", level="error")
            check_res = False
            
        return check_res & super().check()

@dataclass
class BasicSearchConfig(BasicConfig):
    enabling_tot:        bool = False
    gs_eval_batch_size:   int = 1
    gs_gen_batch_size:    int = 1
    gs_gen_repeat_times:  int = 2
    gs_breadth:           int = 2
    gs_iterations:        int = 2

    enabling_bon:        bool = False
    best_of_n_times:      int = 1
    min_step_lengths:     int = 1


    def update(self, args: dict):
        return BasicConfig.update(self, args, "generation_search")
    
    def check(self):
        
        # TODO:
        check_res = True
        return check_res & super().check()


