# encoding=utf-8
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from .base import (BasicConfig, BasicGenerateConfig, AVAILABLE_LR_SCHEDULER,
                   BasicMegatronDeepspeedConfig, BasicTrainerConfig)
from .data import DataConfig
from .logger import LoggerConfig
from .model import PolicyModelConfig, RewardModelConfig, ReflectModelConfig
from chatgpt.utils import logging_rank_0


DEFAULT_DATA_SPLIT_RATIO = {
    "train": 6,
    "eval": 2,
    "test": 2
}

DEFAULT_SAVE_BEST_N_CKPT = 3


@dataclass
class BasicRMTrainerConfig(BasicTrainerConfig, BasicMegatronDeepspeedConfig, LoggerConfig, DataConfig, RewardModelConfig, PolicyModelConfig):
    
    num_workers:                int                         = 2
    total_steps:                Optional[int]               = None
    rm_batch_size:              int                         = 1
    val_check_inter_val:        float                       = 0.05
    max_epochs:                 int                         = 1
    from_ckpt:                  Optional[str]               = None
    activation_checkpointing:   bool                        = False
    data_split_ratio:           Optional[Dict[str,float]]   = None
    save_splited_dataset:       bool                        = True
    val_check_interval:         float                       = 0.05
    save_best_n_ckpt:           int                         = DEFAULT_SAVE_BEST_N_CKPT
    from_sft:                   bool                        = False
    
    def update(self, args: dict):
        BasicTrainerConfig.update(self, args)
        BasicMegatronDeepspeedConfig.update(self, args)
        LoggerConfig.update(self, args)
        DataConfig.update(self, args)
        RewardModelConfig.update(self, args)
        PolicyModelConfig.update(self, args)
        
        return BasicConfig.update(self, args, "reward_modeling")
    
    
    
    def check(self):
        
        check_res = True
        
        if self.data_split_ratio is None:
            logging_rank_0(msg=f"Use default data_split_ratio ({DEFAULT_DATA_SPLIT_RATIO})", level="warning")
            self.data_split_ratio = DEFAULT_DATA_SPLIT_RATIO
        elif not isinstance(self.data_split_ratio, dict):
            logging_rank_0(msg=f"Detected wrong type on data_split_ratio ({type(self.data_split_ratio)}), use default ({DEFAULT_DATA_SPLIT_RATIO})", level="warning")
            self.data_split_ratio = DEFAULT_DATA_SPLIT_RATIO
        else:
            train = self.data_split_ratio.get("train", None)
            eval = self.data_split_ratio.get("eval", None)
            test = self.data_split_ratio.get("test", None)
            if train is None:
                logging_rank_0(msg=f"Use default data_split_ratio/train ({DEFAULT_DATA_SPLIT_RATIO['train']}).", level="warning")
                train = DEFAULT_DATA_SPLIT_RATIO["train"]
            elif not isinstance(train, float) and not isinstance(train, int):
                logging_rank_0(msg=f"Detected wrong type on data_split_ratio/train ({type(self.train)}), which should be 'int' or 'float'. Use default ({DEFAULT_DATA_SPLIT_RATIO['train']})", level="warning")
                train = DEFAULT_DATA_SPLIT_RATIO["train"]
            if eval is None:
                logging_rank_0(msg=f"Use default data_split_ratio/eval ({DEFAULT_DATA_SPLIT_RATIO['eval']}).", level="warning")
                eval = DEFAULT_DATA_SPLIT_RATIO["eval"]
            elif not isinstance(eval, float) and not isinstance(eval, int):
                logging_rank_0(msg=f"Detected wrong type on data_split_ratio/eval ({type(self.eval)}), which should be 'int' or 'float'. Use default ({DEFAULT_DATA_SPLIT_RATIO['eval']})", level="warning")
                eval = DEFAULT_DATA_SPLIT_RATIO["eval"]
            if test is None:
                logging_rank_0(msg=f"Use default data_split_ratio/test ({DEFAULT_DATA_SPLIT_RATIO['test']}).", level="warning")
                test = DEFAULT_DATA_SPLIT_RATIO["test"]
            elif not isinstance(test, float) and not isinstance(test, int):
                logging_rank_0(msg=f"Detected wrong type on data_split_ratio/test ({type(self.test)}), which should be 'int' or 'float'. Use default ({DEFAULT_DATA_SPLIT_RATIO['test']})", level="warning")
                test = DEFAULT_DATA_SPLIT_RATIO["test"]
            
            total = train + eval + test
            self.data_split_ratio = {
                "train": train / total,
                "eval": eval / total,
                "test": test / total
            }
        
        if not isinstance(self.save_best_n_ckpt, int) or self.save_best_n_ckpt <= 0:
            logging_rank_0(msg=f"save_best_n_ckpt ({self.save_best_n_ckpt}) should be greater than 0. Use detault ({DEFAULT_SAVE_BEST_N_CKPT}).", level="warning")
            self.save_best_n_ckpt = DEFAULT_SAVE_BEST_N_CKPT
        
        return check_res \
            & BasicTrainerConfig.check(self) \
            & BasicMegatronDeepspeedConfig.check(self) \
            & LoggerConfig.check(self) \
            & DataConfig.check(self) \
            & RewardModelConfig.check(self)
    

@dataclass
class BasicPPOTrainerConfig(
    BasicTrainerConfig,
    BasicMegatronDeepspeedConfig,
    LoggerConfig, DataConfig,
    RewardModelConfig,
    BasicGenerateConfig,
    PolicyModelConfig):
    """ PPO基础流程（Token-level + Guide + task）"""
    
    critic_from_sft:bool = True
    
    # Batch size
    experience_batch_size:      int     = 128
    generate_minibatch_size:    int     = 32
    policy_minibatch_size:      int     = 8
    rm_minibatch_size:          int     = 2
    sample_batch_size:          int     = 128
    buffer_limit_size:          int     = 512
    replay_buffer_cpu_offload:  bool    = True
    sample_replay_buffer:       bool    = False
    
    # PPO param
    eps_clip:                   float   = 0.2
    value_clip:                 float   = 0.2
    drop_approx_kl:             float   = 0.001
    gamma:                      float   = 0.99
    lam:                        float   = 0.95
    
    # PPO mode
    enable_reward_scaling:              bool    = True
    enable_constrain_actor:             bool    = False
    update_constrain_actor_interval:    int     = 0
    constrain_actor_kl_coef:            float   = 0.01
    target_constrain_actor_kl:          Optional[float] = None,
    kl_adaptor_horizon:                 Optional[float] = None,

    
    # ppopp param
    ppopp_beta:                 float   = 0.5
    ppopp_beta_decay:           float   = 1.0
    ppopp_rate:                 float   = 0.9
    ppopp_rate_decay:           float   = 0.98
    use_guide_action:           bool    = False
    
    # trainer param
    num_workers:                int     = 2
    num_episodes:               int     = 512
    total_steps:                int     = 512
    max_epoch_per_update:       int     = 2
    actor_lr:                   float   = 2e-6
    critic_lr:                  float   = 1e-5
    actor_scheduler_type:       Optional[str] = None
    critic_scheduler_type:      Optional[str] = None
    policy_train_batch_size:    int     = 2
    
    # other trainer param
    update_timesteps:           int     = 1
    max_timesteps:              int     = 1
    
    do_validation:              bool    = True
    val_every_n_episode:        int     = 5
    val_size_per_task:          int     = 16
    clip_grad:                  bool    = True
    
    # coef
    kl_coef:                    float   = 0.0
    entropy_loss_coef:          float   = 0.01
    entropy_loss_decay_rate:    float   = 1.0
    
    # lora param
    enable_policy_lora:         bool    = False
    enable_rm_lora:             bool    = False
    lora_alpha:                 int     = 64
    lora_rank:                  int     = 16
    lora_dropout:              float   = 0.05
    
    def update(self, args):
        BasicTrainerConfig.update(self, args)
        BasicMegatronDeepspeedConfig.update(self, args)
        LoggerConfig.update(self, args)
        DataConfig.update(self, args)
        RewardModelConfig.update(self, args)
        BasicGenerateConfig.update(self, args)
        PolicyModelConfig.update(self, args)
        
        ppo_namespace = args.get("ppo", None)
        if ppo_namespace is None:
            return
        
        BasicConfig.update(self, args["ppo"], "experience")
        BasicConfig.update(self, args["ppo"], "ppo_details")
        BasicConfig.update(self, args["ppo"], "ppopp_details")
        BasicConfig.update(self, args["ppo"], "trainer")
        return

    def check(self):
        
        check_res = BasicTrainerConfig.check(self) \
                & BasicMegatronDeepspeedConfig.check(self) \
                & LoggerConfig.check(self) \
                & DataConfig.check(self) \
                & RewardModelConfig.check(self) \
                & BasicGenerateConfig.check(self) \
                & PolicyModelConfig.check(self)
        
        if not check_res:
            return check_res
        
        if self.actor_scheduler_type is None:
            logging_rank_0(msg=f"Set actor_scheduler_type to detault ({self.scheduler_type}).", level="info")
            self.actor_scheduler_type = self.scheduler_type
        elif self.actor_scheduler_type not in AVAILABLE_LR_SCHEDULER:
            logging_rank_0(msg=f"actor_scheduler_type ({self.actor_scheduler_type}) should be in ({AVAILABLE_LR_SCHEDULER}). Set to detault ({self.scheduler_type}).", level="warning")
            self.actor_scheduler_type = self.scheduler_type
            
        if self.critic_scheduler_type is None:
            logging_rank_0(msg=f"Set critic_scheduler_type to detault ({self.scheduler_type}).", level="info")
            self.critic_scheduler_type = self.scheduler_type
        elif self.critic_scheduler_type not in AVAILABLE_LR_SCHEDULER:
            logging_rank_0(msg=f"critic_scheduler_type ({self.critic_scheduler_type}) should be in ({AVAILABLE_LR_SCHEDULER}). Set to detault ({self.scheduler_type}).", level="warning")
            self.critic_scheduler_type = self.scheduler_type
            
        # 检查lora配置
        if self.enable_policy_lora:
            if self.lora_rank <= 0:
                logging_rank_0(msg=f"lora_rank ({self.lora_rank}) should be greater than zero. Disable lora training.", level="warning")
                self.enable_policy_lora = False
            if not self.critic_from_sft:
                logging_rank_0(msg=f"critic_from_sft ({self.critic_from_sft}) should be 'True' when enabling policy lora. Set to 'True'.", level="warning")
                self.critic_from_sft = True
        
        # TODO: 补充参数检查逻辑
        # if not self.enable_token_reward and self.enable_mixed_sample_reward:
        #     logging_rank_0(msg=f"'enable_mixed_sample_reward' requires 'enable_token_reward'.Setting 'enable_mixed_sample_reward' to False.", level="warning")
        #     self.enable_mixed_sample_reward = False
            
        return check_res
                

@dataclass
class BasicEDPOTrainerConfig(
    BasicTrainerConfig,
    BasicMegatronDeepspeedConfig,
    LoggerConfig, DataConfig,
    RewardModelConfig,
    BasicGenerateConfig,
    PolicyModelConfig):
    """ EPPO """

    # experience
    experience_batch_size:      int     = 128
    generate_minibatch_size:    int     = 32
    policy_minibatch_size:      int     = 8
    rm_minibatch_size:          int     = 2
    sample_batch_size:          int     = 128
    buffer_limit_size:          int     = 512
    replay_buffer_cpu_offload:  bool    = True
    sample_replay_buffer:       bool    = False

    # edpo
    equalizing_preferences:     bool    = False
    max_n_preferences:          int     = 3
    dpo_beta:                   float   = 0.5
    has_ref_model_constraints:  bool    = True
    edpo_preference_batch_size: int     = 1
    ignore_ref_first_n_steps:   int     = -1
    save_every_n_episode:       int     = 5
    sample_replay_buffer:       bool    = False

    # trainer param
    num_workers:                int     = 2
    num_episodes:               int     = 512
    total_steps:                int     = 512
    max_epoch_per_update:       int     = 2
    actor_lr:                   float   = 2e-6
    critic_lr:                  float   = 1e-5
    policy_train_batch_size:    int     = 2
    activation_checkpointing:   bool    = False

    # other trainer param
    update_timesteps:           int     = 1
    max_timesteps:              int     = 1

    do_validation:              bool    = True
    val_every_n_episode:        int     = 5
    val_size_per_task:          int     = 16
    clip_grad:                  bool    = True

    
    def update(self, args):
        BasicTrainerConfig.update(self, args)
        BasicMegatronDeepspeedConfig.update(self, args)
        LoggerConfig.update(self, args)
        DataConfig.update(self, args)
        RewardModelConfig.update(self, args)
        BasicGenerateConfig.update(self, args)
        PolicyModelConfig.update(self, args)

        edpo_namespace = args.get("edpo", None)
        if edpo_namespace is None:
            return

        BasicConfig.update(self, args["edpo"], "experience")
        BasicConfig.update(self, args["edpo"], "edpo_details")
        BasicConfig.update(self, args["edpo"], "trainer")
        return

    def check(self):
        
        check_res = True

        if self.dpo_beta < 0 or self.dpo_beta > 0.5:
            logging_rank_0(msg=f"Unknown behavior using dpo_beta(= {self.dpo_beta})", level="warning")
        
        if self.equalizing_preferences and self.max_n_preferences < 2:
            logging_rank_0(msg=f"EDPO requires at least 2 preferences ({self.max_n_preferences} is give)", level="error")

            
        return check_res \
                & BasicTrainerConfig.check(self) \
                & BasicMegatronDeepspeedConfig.check(self) \
                & LoggerConfig.check(self) \
                & DataConfig.check(self) \
                & RewardModelConfig.check(self) \
                & BasicGenerateConfig.check(self) \
                & PolicyModelConfig.check(self)
                

@dataclass
class BasicFDTrainerConfig(
    BasicTrainerConfig, 
    BasicMegatronDeepspeedConfig,
    BasicGenerateConfig,
    LoggerConfig, DataConfig,
    PolicyModelConfig,
    ReflectModelConfig):


    ### Dataset param
    dataset_train_ratio:        float               = 0.9
    num_workers:                int                 = 2

    ### batch size
    actor_train_batch_size:     int                 = 2         # dataloader
    experience_batch_size:      int                 = 128       # experience_maker
    actor_mini_batch_size:      int                 = 8         # experience_maker
    reflector_mini_batch_size:  int                 = 8         # experience_maker
    sample_batch_size:          int                 = 128       # replay_buffer
    buffer_limit_size:          int                 = 512       # replay_buffer
    replay_buffer_cpu_offload:  bool                = True      # replay_buffer
    sample_replay_buffer:       bool                = False     # replay_buffer

    ### training param
    actor_lr:                   float               = 1e-6
    actor_scheduler_type:       Optional[str]       = None
    max_epochs:                 int                 = 1
    num_episodes:               int                 = 512
    update_timesteps:           int                 = 1
    max_timesteps:              int                 = 1
    val_every_n_episode:        int                 = 5
    total_steps:                int                 = 512
    skip_exp:                   bool                = False

    ### FD param
    divergence_type:            str                 = "JSD"    
    JSD_coef:                   float               = 0.5
    GKD_coef:                   float               = 1.0
    KD_temperature:             float               = 1.0
    shrink:                     Tuple[int]          = (0, 0)
    level:                      str                 = "token"

    ### actor gen args
    actor_do_sample:            bool                = True
    actor_top_p:                float               = 0.85
    actor_top_k:                int                 = 0
    actor_repetition_penalty:   float               = 1.0
    actor_temperature:          float               = 0.85
    actor_max_new_tokens:       int                 = 2048

    ### reflector gen args
    reflector_do_sample:            bool                = True
    reflector_top_p:                float               = 0.85
    reflector_top_k:                int                 = 0
    reflector_repetition_penalty:   float               = 1.0
    reflector_temperature:          float               = 0.85
    reflector_max_new_tokens:       int                 = 2048

    def update(self, args):
        BasicTrainerConfig.update(self, args)
        BasicMegatronDeepspeedConfig.update(self, args)
        LoggerConfig.update(self, args)
        DataConfig.update(self, args)
        ReflectModelConfig.update(self, args)
        BasicGenerateConfig.update(self, args)
        PolicyModelConfig.update(self, args)
        
        FD_namespace = args.get("FD", None)
        if FD_namespace is None:
            return
        if isinstance(FD_namespace, dict):
            BasicConfig.update(self, FD_namespace, "experience")
            BasicConfig.update(self, FD_namespace, "loss")
            BasicConfig.update(self, FD_namespace, "trainer")
            BasicConfig.update(self, FD_namespace, "actor_gen_args")
            BasicConfig.update(self, FD_namespace, "reflector_gen_args")
        return

    def check(self):

        check_res = True

        self.level = self.level.lower()
        assert self.level in ['sequence', 'token']

        if self.actor_scheduler_type is None:
            logging_rank_0(msg=f"Set actor_scheduler_type to detault ({self.scheduler_type}).", level="info")
            self.actor_scheduler_type = self.scheduler_type
            
        return check_res \
                & BasicTrainerConfig.check(self) \
                & BasicMegatronDeepspeedConfig.check(self) \
                & LoggerConfig.check(self) \
                & DataConfig.check(self) \
                & ReflectModelConfig.check(self) \
                & BasicGenerateConfig.check(self) \
                & PolicyModelConfig.check(self)