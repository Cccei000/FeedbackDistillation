# encoding=utf-8
import os
from enum import Enum, unique #, Enum
# from strenum import StrEnum
from typing import Dict, Optional

import yaml

from chatgpt.utils import logging_rank_0

AVAILABLE_TP = [1,2,4,8]
AVAILABLE_PP = [1,2,4,8]

# 可选的生成模型结构类型：
#   [1] llama_13B
#   [2] llama2_13B
AVAILABLE_POLICY_MODEL_TYPE = ["llama_13B", "llama2_13B", "baichuan_13B"]

@unique
class PolicyModelType(str,Enum):
    """可选的生成模型结构类型：\\
        [llama_13B]\\
        [llama2_13B]\\
        [baichuan_13B]
    """    
    llama_13B = AVAILABLE_POLICY_MODEL_TYPE[0]
    llama2_13B = AVAILABLE_POLICY_MODEL_TYPE[1]
    baichuan_13B = AVAILABLE_POLICY_MODEL_TYPE[2]

# 
#   [1] llama_13B
AVAILABLE_REWARD_MODEL_TYPE = ["llama_13B", "llama2_13B"]

@unique
class RewardModelType(str,Enum):
    """可选的奖励模型结构类型：\\
        [llama_13B]\\
    """
    llama_13B = AVAILABLE_REWARD_MODEL_TYPE[0]
    llama2_13B = AVAILABLE_REWARD_MODEL_TYPE[1]
    
AVAILABLE_FEEDBACK_MODEL_TYPE = ["auto-j_13B", "ultraCM_13B"]

AVAILABLE_REWARD_MODEL_GRANULARITY = ["token", "sample", "token_mix_sample"]

@unique
class RewardModelGranularity(str,Enum):
    """奖励模型训练采用的粒度:\\
        [token] 单头token level奖励模型\\
        [sample] 单头sample level奖励模型\\
        [token_mix_sample] 双头的奖励模型（value_head为sample level，token_value_head为token level）
    """
    token = AVAILABLE_REWARD_MODEL_GRANULARITY[0]
    sample = AVAILABLE_REWARD_MODEL_GRANULARITY[1]
    token_mix_sample = AVAILABLE_REWARD_MODEL_GRANULARITY[2]

AVAILABLE_ACTOR_GRANULARITY = ["token", "step", "sample"]

@unique
class ActorGranularity(str,Enum):
    """PPO训练采用的粒度\\
        [token]\\
        [step]\\
        [sample]
    """
    token = AVAILABLE_ACTOR_GRANULARITY[0]
    step = AVAILABLE_ACTOR_GRANULARITY[1]
    sample = AVAILABLE_ACTOR_GRANULARITY[2]

# Logger信息等级
AVAILABLE_LOGGING_LEVEL = ["debug", "info"]

@unique
class LoggingLevel(str,Enum):
    debug = AVAILABLE_LOGGING_LEVEL[0]
    info = AVAILABLE_LOGGING_LEVEL[1]
    
# 支持的流程类型
PIPELINE = ["reward_modeling", "ppo", "edpo", "prepare"]

@unique
class PipelineName(str,Enum):
    reward_modeling = PIPELINE[0]
    ppo = PIPELINE[1]
    edpo = PIPELINE[2]
    prepare = PIPELINE[3]
    
# prepare流程中需要准备的模型
AVAILABLE_PREPARE_LIST = ['policy', 'rm']


@unique
class DeepspeedStage(str,Enum):
    stage_1 = "stage_1"
    stage_2_offload = "stage_2_offload"

# FIXME: 修改为最终确定的路径
DEFAULT_POLICY_MODEL_PATH = {
    "llama_13B": "/cognitive_comp/zhangwenjun/checkpoints/llama-neox-sft/merged_0630/merged_average-chat_19000-mmm_0615_ind_chat_19000_math_6000-mer_0619_ind_chat_19000_18000_math_6000",
    "llama2_13B": "/cognitive_comp/songzhuoyang/models/llama2_13B_sft/step_26k_hf",
    "baichuan_13B": "/cognitive_comp/songzhuoyang/models/baichuan2_13B/local",
}
DEFAULT_REWARD_MODEL_PATH = {
    "llama_13B": {
        "token": "/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0725_token/global_step7202_hf",
        "sample": "/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0725/global_step8349_hf",
        "token_mix_sample": "/cognitive_comp/liangyuxin/workspace/pipeline/ckpt/reward_model/0817_RM13B_MIX/global_step6517_hf",
    },
    "llama2_13B": {
        "sample": "/cognitive_comp/songzhuoyang/models/llama2_13B_rm/0829_14k",
        #TODO add proper path
        "token": "/cognitive_comp/songzhuoyang/models/llama2_13B_rm/0829_14k",
        "token_mix_sample": "/cognitive_comp/songzhuoyang/models/llama2_13B_rm/0829_14k",
    }
}
DEFAULT_TOEKNIZER_PATH = {
    "llama_13B": "/cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-c-pretrain-tokenizer/",
    "llama2_13B": "/cognitive_comp/songzhuoyang/models/llama2_13B_sft/step_26k_hf",
    "baichuan_13B": "/cognitive_comp/songzhuoyang/models/baichuan2_13B/local",
}

# TODO：补充各训练流程的配置文件
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
DEFAULT_PPO_SCRIPT_PATH:Dict[ActorGranularity,Dict[RewardModelGranularity,str]] = {
    ActorGranularity.token: {
        RewardModelGranularity.token:               os.path.join(SCRIPT_DIR, "ppo_token_level.yml"),
        RewardModelGranularity.token_mix_sample:    os.path.join(SCRIPT_DIR, "ppo_token_level.yml"),
        RewardModelGranularity.sample:              os.path.join(SCRIPT_DIR, "ppo_token_level.yml"),
    },
    ActorGranularity.sample: {
        RewardModelGranularity.token:               os.path.join(SCRIPT_DIR, "ppo_sample_level.yml"),
        RewardModelGranularity.token_mix_sample:    os.path.join(SCRIPT_DIR, "ppo_sample_level.yml"),
        RewardModelGranularity.sample:              os.path.join(SCRIPT_DIR, "ppo_sample_level.yml"),
    },
    ActorGranularity.step: {
        RewardModelGranularity.token:               os.path.join(SCRIPT_DIR, "ppo_step_level.yml"),
        RewardModelGranularity.token_mix_sample:    os.path.join(SCRIPT_DIR, "ppo_step_level.yml"),
        RewardModelGranularity.sample:              os.path.join(SCRIPT_DIR, "ppo_step_level.yml"),
    }
}
DEFAULT_RM_TRAINING_SCRIPT_PATH:Dict[RewardModelGranularity,str] = {
    RewardModelGranularity.token: os.path.join(SCRIPT_DIR, "token_level_full_pipeline_all_param.yml"),
    RewardModelGranularity.token_mix_sample: os.path.join(SCRIPT_DIR, "token_level_full_pipeline_all_param.yml"),
    RewardModelGranularity.sample: os.path.join(SCRIPT_DIR, "token_level_full_pipeline_all_param.yml")
}

DEFAULT_EDPO_SCRIPT_PATH:str = os.path.join(SCRIPT_DIR, "edpo.yml")

DEFAULT_SCRIPT_PATH:str = os.path.join(SCRIPT_DIR, "defaults.yml")

DEFAULT_FD_SCRIPT_PATH:str = os.path.join(SCRIPT_DIR, "FD.yml")

DEFAULT_DEEPSPEED_SCRIPT_PATH:str = {
    DeepspeedStage.stage_1: os.path.join(SCRIPT_DIR, "deepspeed_stage_1.yml"),
    DeepspeedStage.stage_2_offload: os.path.join(SCRIPT_DIR, "deepspeed_stage_2_offload.yml")
}




def assemble_pipeline_scripts(available_gpus:int, pipeline_script_path:Optional[int]=None) -> dict:
    """根据用户的pipeline配置，组装默认配置

    Args:
        available_gpus (dict): 可用的gpu数量

    Returns:
        dict: 默认配置
    """
    config = {}
    # defaults
    with open(DEFAULT_SCRIPT_PATH, "r") as f:
        config.update(yaml.load(f, Loader=yaml.FullLoader))
        
    # megatron （默认按照A100 80G配置）
    if available_gpus % 8 == 0:
        stage = DeepspeedStage.stage_1
        tp_size = 8
    elif available_gpus % 4 == 0:
        stage = DeepspeedStage.stage_1 if available_gpus >= 8 else DeepspeedStage.stage_2_offload
        tp_size = 4
    elif available_gpus % 2 == 0:
        stage = DeepspeedStage.stage_1 if available_gpus >= 8 else DeepspeedStage.stage_2_offload
        tp_size = 2
    else:
        stage = DeepspeedStage.stage_2_offload
        tp_size = 1
    pp_size = 1
        
    with open(DEFAULT_DEEPSPEED_SCRIPT_PATH[stage], "r") as f:
        config.update(yaml.load(f, Loader=yaml.FullLoader))
    config["megatron"]["tensor_model_parallel_size"] = tp_size
    config["megatron"]["pipe_model_parallel_size"] = pp_size
    
    logging_rank_0(f"Enable Deepspeed_{stage.value} with tensorparallel size '{tp_size}' and pipeline parallel size '{pp_size}'.", "debug")
    
    if pipeline_script_path:
        # 流程相关的默认配置
        with open(pipeline_script_path, "r") as f:
            config.update(yaml.load(f, Loader=yaml.FullLoader))
    return config
