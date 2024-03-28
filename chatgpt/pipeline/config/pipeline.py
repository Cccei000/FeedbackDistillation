# encoding=utf-8
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

from chatgpt.utils import logging_rank_0

from .base import BasicConfig, BasicSearchConfig
from .config import (AVAILABLE_LOGGING_LEVEL, AVAILABLE_POLICY_MODEL_TYPE,
                     AVAILABLE_ACTOR_GRANULARITY, AVAILABLE_PREPARE_LIST,
                     AVAILABLE_FEEDBACK_MODEL_TYPE,
                     AVAILABLE_REWARD_MODEL_GRANULARITY,
                     AVAILABLE_REWARD_MODEL_TYPE, ActorGranularity, RewardModelGranularity,
                     DEFAULT_EDPO_SCRIPT_PATH, DEFAULT_POLICY_MODEL_PATH,
                     DEFAULT_PPO_SCRIPT_PATH, DEFAULT_REWARD_MODEL_PATH,
                     DEFAULT_RM_TRAINING_SCRIPT_PATH, DEFAULT_TOEKNIZER_PATH,
                     DEFAULT_FD_SCRIPT_PATH,
                     assemble_pipeline_scripts)
from .model import ModelConvertConfig
from .trainer import (BasicEDPOTrainerConfig, BasicPPOTrainerConfig,
                      BasicRMTrainerConfig, BasicFDTrainerConfig)


@dataclass
class PipelineConfig(BasicConfig):

    policy_model_type:  str = AVAILABLE_POLICY_MODEL_TYPE[0]
    reward_model_type:  str = AVAILABLE_REWARD_MODEL_TYPE[0]
    reflector_model_type: Optional[str] = None
    pipeline:           Optional[List[str]] = None
    prepare_list:       Optional[List[str]] = None
    rm_granularity:     Optional[str] = None
    actor_granularity:    Optional[str] = None
    dataset_path:       Optional[str] = None
    workspace_path:     Optional[str] = None
    logging_path:       Optional[str] = None
    logging_level:      str = "info"
    gpus:               int = 8
    enable_flops_profiler: bool = False

    def update(self, args: dict):
        BasicConfig.update(self, args, "pipeline")
        return 
    
    def check(self):
        
        check_res = True
        # 检查 logging level
        if self.logging_level not in AVAILABLE_LOGGING_LEVEL:
            logging_rank_0(msg=f"logging_level ({self.logging_level}) should be in {set(AVAILABLE_LOGGING_LEVEL)}, set to 'info'.", level="warning")
        
        # ppo || edpo || prepare policy model：检查 policy 模型的 type 是否合法
        if "ppo" in self.pipeline or "edpo" in self.pipeline or (self.prepare_list and "policy" in self.prepare_list):
            if self.policy_model_type is None:
                logging_rank_0(msg=f"policy_model_type is required.", level="error")
                check_res = False 
            elif self.policy_model_type not in AVAILABLE_POLICY_MODEL_TYPE:
                logging_rank_0(f"policy_model_type ({self.policy_model_type}) should in {set(AVAILABLE_POLICY_MODEL_TYPE)}.")
                check_res = False 
        
        # rm 训练 || ppo || prepare rm：检查 rm 的 type 是否合法
        if "reward_modeling" in self.pipeline or "ppo" in self.pipeline or (self.prepare_list and 'rm' in self.prepare_list):
            if self.reward_model_type is None:
                logging_rank_0(msg=f"policy_model_type is required ({set(AVAILABLE_REWARD_MODEL_TYPE)}).", level="error")
                check_res = False
            elif self.reward_model_type not in AVAILABLE_REWARD_MODEL_TYPE:
                logging_rank_0(msg=f"reward_model_type ({self.reward_model_type}) should in {set(AVAILABLE_REWARD_MODEL_TYPE)}.", level="error")
                check_res = False
        
        # 检查 prepare list 中声明的模型是否合法
        if self.prepare_list  and not set(self.prepare_list).issubset(set(AVAILABLE_PREPARE_LIST)):
            logging_rank_0(msg=f"prepare_list ({self.prepare_list}) should in {set(AVAILABLE_PREPARE_LIST)}.", level="error")
            check_res = False

        # 检查数据集路径是否存在
        if self.dataset_path is None:
            logging_rank_0(msg=f"dataset_path is required.", level="error")
            check_res = False
        elif not os.path.exists(self.dataset_path):
            logging_rank_0(msg=f"dataset_path ({self.dataset_path}) isn't a directory.", level="error")
            check_res = False
        
        # 检查workspace路径是否合法
        if self.workspace_path is None:
            logging_rank_0(msg=f"workspace_path is required.", level="error")
            check_res = False
        elif not os.path.exists(self.workspace_path):
            logging_rank_0(msg=f"workspace_path ({self.workspace_path}) doesn't exist. Make it.", level="warning")
            os.makedirs(self.workspace_path, exist_ok=True)
        
        if self.logging_path is None:
            self.logging_path = self.workspace_path
        elif not os.path.exists(self.logging_path):
            logging_rank_0(msg=f"logging_path ({self.logging_path}) doesn't exist. Setting to workspace_path ({self.workspace_path}).", level="warning")
            self.logging_path = self.workspace_path

        if "ppo" in self.pipeline \
            or "reward_modeling" in self.pipeline \
                or (self.prepare_list and "prepare" in self.pipeline and "rm" in self.prepare_list):
            if self.rm_granularity is None or self.rm_granularity not in AVAILABLE_REWARD_MODEL_GRANULARITY:
                logging_rank_0(msg=f"rm_granularity is required and should be in {set(AVAILABLE_REWARD_MODEL_GRANULARITY)}.", level="error")
                check_res = False
        
        if "ppo" in self.pipeline \
            or (self.prepare_list and "prepare" in self.pipeline and "policy" in self.prepare_list):
            if self.actor_granularity is None or self.actor_granularity not in AVAILABLE_ACTOR_GRANULARITY:
                logging_rank_0(msg=f"actor_granularity is required and should be in {set(AVAILABLE_ACTOR_GRANULARITY)}.", level="error")
                check_res = False
            
        return check_res & super().check()


@dataclass
class ModelConvertPipelineConfig(PipelineConfig, ModelConvertConfig):
    def update(self, args: dict):
        ## PipelineConfig的优先级高于ModelConvertConfig
        ## 先使用ModelConvertConfig更新，再使用PipelineConfig更新
        ModelConvertConfig.update(self, args)
        return super().update(args)
    
    def update_default_config(self, config:PipelineConfig):
        """基于基础流程配置初始化各配置

        Args:
            config (PipelineConfig): _description_
        """
        
        pipeline = config.pipeline
        gpus = config.gpus
        assert "prepare" in pipeline
        # 读取模型转换的基础配置
        self.update(args=assemble_pipeline_scripts(pipeline_script_path=None, available_gpus=gpus))
        
        return 
    
    def check(self):
        check_res = PipelineConfig.check(self) & ModelConvertConfig.check(self)
        
        if not check_res:
            return False
    
        ## 检查model_path和tokenizer_path，并基于model_type设置默认值
        if "ppo" in self.pipeline or (self.prepare_list and 'policy' in self.prepare_list):
            default_path:str = DEFAULT_POLICY_MODEL_PATH.get(self.policy_model_type, None) if self.policy_model_type else None
            self.policy_model_path = self.check_path(self.policy_model_path, default_path, model_str='policy_model')
            if not self.policy_model_path:
                return False
                    
        # if "reward_modeling" in self.pipeline \
        #     or "ppo" in self.pipeline \
        #         or (self.prepare_list and 'rm' in self.prepare_list):
        if "ppo" in self.pipeline or (self.prepare_list and 'rm' in self.prepare_list):
            default_path: Dict[str,str] = DEFAULT_REWARD_MODEL_PATH.get(self.reward_model_type, None) if self.reward_model_type else None
            default_path: str = default_path.get(self.rm_granularity, None) if default_path else None
            self.reward_model_path = self.check_path(self.reward_model_path, default_path, model_str='reward_model')
            if not self.reward_model_path:
                return False
            
        return True

@dataclass
class PPOPipelineConfig(PipelineConfig, BasicPPOTrainerConfig, BasicSearchConfig):
    
    def update(self, args: dict):
        ## PipelineConfig的优先级高于TrainerConfig
        ## 先使用TrainerConfig更新，再使用PipelineConfig更新
        BasicPPOTrainerConfig.update(self, args)
        BasicSearchConfig.update(self, args)
        return super().update(args)
    
    def update_with_default_config(self, config:PipelineConfig):
        """基于基础流程配置初始化各配置

        Args:
            config (PipelineConfig): _description_
        """
        
        pipeline = config.pipeline
        actor_granularity = config.actor_granularity
        rm_granularity = config.rm_granularity
        gpus = config.gpus
        assert "ppo" in pipeline
        # 根据ppo和rm的粒度，加载默认配置
        self.update(
            args=assemble_pipeline_scripts(
                pipeline_script_path=DEFAULT_PPO_SCRIPT_PATH[actor_granularity][rm_granularity],
                available_gpus=gpus
            )
        )
        
        return 
    
    def check(self):
        check_res = PipelineConfig.check(self) & BasicPPOTrainerConfig.check(self) & BasicSearchConfig.check(self)
        
        if not check_res:
            return False
        
        
        ## 检查model_path和tokenizer_path，并基于model_type设置默认值
        default_path:str = DEFAULT_POLICY_MODEL_PATH.get(self.policy_model_type, None) if self.policy_model_type else None
        self.policy_model_path = self.check_path(self.policy_model_path, default_path, model_str='policy_model')
        if not self.policy_model_path:
            return False

        default_path:Dict[str,str] = DEFAULT_REWARD_MODEL_PATH.get(self.reward_model_type, None) if self.reward_model_type else None
        default_path:str = default_path.get(self.rm_granularity, None) if default_path else None
        self.reward_model_path = self.check_path(self.reward_model_path, default_path, model_str='reward_model')
        if not self.reward_model_path:
            return False

        default_path = DEFAULT_TOEKNIZER_PATH.get(self.policy_model_type, None) if self.policy_model_type else None
        self.policy_tokenizer_path = self.check_path(self.policy_tokenizer_path, default_path, model_str='policy_tokenizer')
        if not self.policy_tokenizer_path:
            return False

        default_path = DEFAULT_TOEKNIZER_PATH.get(self.reward_model_type, None) if self.reward_model_type else None
        self.rm_tokenizer_path = self.check_path(self.rm_tokenizer_path, default_path, model_str='rm_tokenizer')
        if not self.rm_tokenizer_path:
            return False
        
        # FIXME: 模型类型不同时，不支持 Token-level PPO
        if self.policy_model_type != self.reward_model_type and self.actor_granularity == "token":
            logging_rank_0(msg=f"Token-level PPO is not applicable to different model type ('{self.policy_model_type}' policy and '{self.reward_model_type}' rm).", level="error")
            check_res = False
            
        # 检查 bon 和 tot search，只能用于 sample-level 或 step-level ppo
        if self.enabling_bon or self.enabling_tot:
            if self.actor_granularity is ActorGranularity.token:
                logging_rank_0(msg=f"Token-level PPO is not applicable to bon searching and tot searching. Use default generating function.", level="warning")
                self.enabling_bon = False
                self.enabling_tot = False
            
        return True


@dataclass
class RewardModelingPipelineConfig(PipelineConfig, BasicRMTrainerConfig):
    
    def update(self, args: dict):
        ## PipelineConfig的优先级高于TrainerConfig
        ## 先使用TrainerConfig更新，再使用PipelineConfig更新
        BasicRMTrainerConfig.update(self, args)
        return super().update(args)
    
    def update_with_default_config(self, config:PipelineConfig):
        """基于基础流程配置初始化各配置

        Args:
            config (PipelineConfig): _description_
        """
        
        pipeline = config.pipeline
        rm_granularity = config.rm_granularity
        gpus = config.gpus
        assert "reward_modeling" in pipeline
        self.update(
            args=assemble_pipeline_scripts(
                pipeline_script_path=DEFAULT_RM_TRAINING_SCRIPT_PATH[RewardModelGranularity[rm_granularity]],
                available_gpus=gpus
            )
        )
        
        return
    
    def check(self):
        check_res = PipelineConfig.check(self) & BasicRMTrainerConfig.check(self)
        
        if not check_res:
            return False
        
        ## 检查model_path和tokenizer_path，并基于model_type设置默认值
        default_path:Dict[str,str] = DEFAULT_REWARD_MODEL_PATH.get(self.reward_model_type, None) if self.reward_model_type else None
        default_path:str = default_path.get(self.rm_granularity, None) if default_path else None
        self.reward_model_path = self.check_path(self.reward_model_path, default_path, model_str='reward_model')
        if not self.reward_model_path:
            return False

        default_path = DEFAULT_TOEKNIZER_PATH.get(self.reward_model_type, None) if self.reward_model_type else None
        self.rm_tokenizer_path = self.check_path(self.rm_tokenizer_path, default_path, model_str='rm_tokenizer')
        if not self.rm_tokenizer_path:
            return False

        if self.from_sft:
            ## 检查model_path和tokenizer_path，并基于model_type设置默认值
            default_path = DEFAULT_POLICY_MODEL_PATH.get(self.policy_model_type, None) if self.policy_model_type else None
            self.policy_model_path = self.check_path(self.policy_model_path, default_path, model_str='policy_model')
            if not self.policy_model_path:
                return False
            
            default_path = DEFAULT_TOEKNIZER_PATH.get(self.policy_model_type, None) if self.policy_model_type else None
            self.policy_tokenizer_path = self.check_path(self.policy_tokenizer_path, default_path, model_str='policy_tokenizer')
            if not self.policy_tokenizer_path:
                return False

        return True

@dataclass
class EDPOPipelineConfig(PipelineConfig, BasicEDPOTrainerConfig, BasicSearchConfig):
    
    def update(self, args: dict):
        ## PipelineConfig的优先级高于TrainerConfig
        ## 先使用TrainerConfig更新，再使用PipelineConfig更新
        BasicSearchConfig.update(self, args)
        BasicEDPOTrainerConfig.update(self, args)
        return super().update(args)
    
    def update_with_default_config(self, config:PipelineConfig):
        """基于基础流程配置初始化各配置

        Args:
            config (PipelineConfig): _description_
        """
        
        assert "edpo" in config.pipeline
        gpus = config.gpus
        self.update(assemble_pipeline_scripts(available_gpus=gpus, pipeline_script_path=DEFAULT_EDPO_SCRIPT_PATH))
        
        return 
    
    def check(self):
        check_res = PipelineConfig.check(self) & BasicEDPOTrainerConfig.check(self)
        
        if not check_res:
            return check_res
        
        # 如果开启 bon 或 tot 时，需要使用RM
        if self.enabling_bon or self.enabling_tot or self.equalizing_preferences:
            if self.rm_granularity is None or self.rm_granularity not in AVAILABLE_REWARD_MODEL_GRANULARITY:
                logging_rank_0(msg=f"When using reward model, rm_granularity should be in {set(AVAILABLE_REWARD_MODEL_GRANULARITY)}. Or disabling bon, tot and equalizing_preferences.", level="error")
                check_res = False
        
        # EDPO 只支持 Sample-level Actor
        if self.actor_granularity is None:
            self.actor_granularity = ActorGranularity.sample.value
        elif self.actor_granularity != ActorGranularity.sample.value:
            logging_rank_0(msg=f"EDPO is only applicated to sample-level Actor. Set actor_granularity to 'sample'.", level="warning")
            self.actor_granularity = ActorGranularity.sample.value
        
        # EDPO 不支持 Token-level RM
        ## NOTE：这主要是因为action mask 无法传入至 tot_search中，导致无法取ask_mean
        ## FIXME：待tot_search支持action_mask后，去除此限制
        if self.rm_granularity == RewardModelGranularity.token.value:
            logging_rank_0(msg=f"EDPO is currently not applicated to token-level reward model. Please use 'sample-level' or 'token-mix-sample' reward model.", level="error")
            return False
            

        ## 检查model_path和tokenizer_path，并基于model_type设置默认值
        default_path = DEFAULT_POLICY_MODEL_PATH.get(self.policy_model_type, None) if self.policy_model_type else None
        self.policy_model_path = self.check_path(self.policy_model_path, default_path, model_str='policy_model')
        if not self.policy_model_path:
            return False

        default_path = DEFAULT_TOEKNIZER_PATH.get(self.policy_model_type, None) if self.policy_model_type else None
        self.policy_tokenizer_path = self.check_path(self.policy_tokenizer_path, default_path, model_str='policy_tokenizer')
        if not self.policy_tokenizer_path:
            return False

        if self.equalizing_preferences:
            default_path:Dict[str,str] = DEFAULT_REWARD_MODEL_PATH.get(self.reward_model_type, None) if self.reward_model_type else None
            default_path:str = default_path.get(self.rm_granularity, None) if default_path else None
            self.reward_model_path = self.check_path(self.reward_model_path, default_path, model_str='reward_model')
            if not self.reward_model_path:
                return False

            default_path = DEFAULT_TOEKNIZER_PATH.get(self.reward_model_type, None) if self.reward_model_type else None
            self.rm_tokenizer_path = self.check_path(self.rm_tokenizer_path, default_path, model_str='rm_tokenizer')
            if not self.rm_tokenizer_path:
                return False

        return True


@dataclass
class FDPipelineConfig(PipelineConfig, BasicFDTrainerConfig):

    def update(self, args: dict):
        BasicFDTrainerConfig.update(self, args)
        return super().update(args)

    def update_with_default_config(self, config:PipelineConfig):
        assert any(["FD" in step for step in config.pipeline])
        gpus = config.gpus
        self.update(assemble_pipeline_scripts(available_gpus=gpus, pipeline_script_path=DEFAULT_FD_SCRIPT_PATH))
        self.gradient_accumulation_steps = 1
        self.actor_granularity = ActorGranularity.sample.value

    def check(self):
        assert self.reflector_model_type in AVAILABLE_FEEDBACK_MODEL_TYPE or self.reflector_model_type in AVAILABLE_POLICY_MODEL_TYPE
        # assert self.actor_train_batch_size == 1, "This FD pipeline does not support training batchsize > 1 yet."
        return PipelineConfig.check(self) and BasicFDTrainerConfig.check(self)