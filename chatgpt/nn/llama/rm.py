from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fengshen_inner.models.llama.configuration_llama import \
    LlamaConfig as FengshenConfig
from fengshen_inner.models.llama.modeling_llama import \
    LlamaModel as FengshenLlamaModel
from fengshen_inner.models.llama.modeling_llama import LlamaPreTrainedModel
from fengshen_inner.models.megatron import mpu
from transformers import (AutoConfig, AutoModel, LlamaConfig, LlamaModel,
                          PreTrainedModel)

from chatgpt.experience_maker import RunningMeanStd
from chatgpt.logger import Logger
from chatgpt.nn import RewardModel
from chatgpt.nn.utils import masked_mean
from chatgpt.pipeline.config import ActorGranularity, RewardModelGranularity
from chatgpt.utils import CostTimer, LoggingLevel, logging_rank_0


class LlamaRM(RewardModel):
    # RM for inference 
    # value_head is nn.Linear
    def __init__(self,
                 model:nn.Module, 
                 value_head:nn.Module, 
                 rm_granularity:RewardModelGranularity,
                 output_granularity:ActorGranularity,
                 mix_coef:float=0.1,
                 token_value_head:Optional[nn.Module]=None,
                 convert_func:Optional[Callable]=None, 
                 lora_rank:int=0,
                 logger:Optional[Logger]=None) -> None:
        """RM for ppo inference 

        Args:
            model (nn.Module):                                  模型Backbone
            value_head (nn.Module):                             Value head
            rm_granularity (RewardModelGranularity):            RM训练时采用的粒度（决定模型加载的结构和value head推理方式）
            output_granularity (PPOGranularity):                RM输出奖励的粒度
            mix_coef (float, optional):                         对于混合粒度奖励模型，Sample level奖励和Token level奖励混合的系数. Defaults to 0.1.
            token_value_head (Optional[nn.Module], optional):   混合粒度奖励模型的token value head. Defaults to None.
            convert_func (Optional[Callable], optional):        格式转换方法，用于适配生成模型与奖励模型使用不同Prompt或tokenizer的情况. Defaults to None.
            lora_rank (int, optional):                          RM的LoRA rank. Defaults to 0.
            logger (Optional[Logger], optional):                Logger，用于记录RM推理的一些信息. Defaults to None.
        """             
        if lora_rank == 0:
            super().__init__(model, value_head)
        else:
            super().__init__(model, value_head,lora_rank=lora_rank)
        self.convert_func = convert_func
        self.rm_granularity = rm_granularity
        self.output_granularity = output_granularity
        self.enable_checkpointing = False
        self.token_value_head = token_value_head
        if self.rm_granularity==RewardModelGranularity.token_mix_sample and self.token_value_head is None:
            self.token_value_head = mpu.RowParallelLinear(
                    config=model.config,
                    input_size=model.config.hidden_size,
                    output_size=1,
                    input_is_parallel=False,
                    skip_bias_add=False,
                    parallel_output=False,
                )   
        self.mix_coef = mix_coef
        
        self.running_token_level_reward = RunningMeanStd(shape=1, device=self.body.device)
        self.running_sample_level_reward = RunningMeanStd(shape=1, device=self.body.device)
        
        self.enable_reward_aligning = output_granularity == ActorGranularity.token and rm_granularity == RewardModelGranularity.token_mix_sample
        self.logger = logger
        
        #### 粒度检查 ####
        # step-level PPO将token-mix-sample RM当作sample-level RM使用，返回sample-level奖励
        # NOTE: 一般情况下，综合token-mix-sample RM的两个head的输出，效果会更符合预期
        if self.rm_granularity == RewardModelGranularity.token_mix_sample and self.output_granularity == ActorGranularity.step:
                logging_rank_0(f"Step level PPO uses 'token_mix_sample RM' as sample level RM.", LoggingLevel.WARNING)
        
        return
    
    def gradient_checkpointing_enable(self):
        """启动RM的checkpointing
        """        
        self.body.gradient_checkpointing_enable()
        self.enable_checkpointing = True
        return
        
    def forward(self,
                sequences: torch.LongTensor, 
                action_mask: Optional[torch.Tensor]=None,
                attention_mask:  Optional[torch.Tensor]=None,) -> torch.Tensor:  
        """RM推理

        Args:
            sequences (torch.LongTensor):   token ids       (bs, seq_len)
            action_mask (torch.Tensor):     action mask     (bs, seq_len-1)
            attention_mask (torch.Tensor):  attention mask  (bs, seq_len)

        Returns:
            torch.Tensor:                   reward          (bs,) or (bs, seq_len)
        """      
        if attention_mask is None:
            attention_mask = torch.ones_like(sequences)
        # 不同模型之间的prompt、vocab的转换
        if self.convert_func is not None:
            if action_mask is None:
                sequences, attention_mask, _ = self.convert_func(sequences, attention_mask, action_mask, self.body.device)
            else:
                sequences, attention_mask, action_mask = self.convert_func(sequences, attention_mask, action_mask, self.body.device)
        
        # 计算RM forward时间
        with CostTimer():
            
            # Backbone infer
            outputs = self.body(sequences, attention_mask=attention_mask, output_hidden_states=True, use_cache=True)
            last_hidden_states = outputs.hidden_states[-1]
        
            if self.rm_granularity == RewardModelGranularity.token:
                return self._forward_token_rm(last_hidden_states=last_hidden_states, action_mask=action_mask, attention_mask=attention_mask)
            elif self.rm_granularity == RewardModelGranularity.sample:
                return self._forward_sample_rm(last_hidden_states=last_hidden_states, action_mask=action_mask, attention_mask=attention_mask)
            else:
                return self._forward_token_mix_rm(last_hidden_states=last_hidden_states, action_mask=action_mask, attention_mask=attention_mask)
    
    def _forward_token_rm(self,
                                 last_hidden_states: torch.Tensor, 
                                 action_mask: torch.Tensor,
                                 attention_mask: torch.Tensor) -> torch.Tensor:
        """token level RM 推理

        Args:
            last_hidden_states (torch.Tensor):  LM 输出的 last hidden states
            action_mask (torch.Tensor):         action mask
            attention_mask (torch.Tensor):      attention mask

        Returns:
            torch.Tensor: _description_
        """            
        # Value head forward
        values = self.value_head(last_hidden_states)[0].squeeze(-1) #  (bs,len_seq)
        
        # step-level & sample-level PPO：对token-level RM的value_head的输出进行平均，作为奖励
        if self.output_granularity == ActorGranularity.step or self.output_granularity == ActorGranularity.sample:
            action_mask = F.pad(action_mask, (1, 0), value=False)  # make_exp 返回action_mask为(bs, len_seq) 
            values = masked_mean(values,action_mask) # (bs,)
            return values
        
        # token-level PPO：使用token-level RM的value_head的输出作为奖励
        return values
    
    def _forward_sample_rm(self,
                                  last_hidden_states: torch.Tensor,
                                  action_mask: torch.Tensor,
                                  attention_mask: torch.Tensor) -> torch.Tensor:
        """sample level RM 推理

        Args:
            last_hidden_states (torch.Tensor):  LM 输出的 last hidden states
            action_mask (torch.Tensor):         action mask
            attention_mask (torch.Tensor):      attention mask

        Returns:
            torch.Tensor: _description_
        """
        # Value head forward，取last token value
        last_index =  torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in attention_mask], dtype=torch.int64)
        last_hidden_states = last_hidden_states[torch.arange(last_hidden_states.shape[0]), last_index]
        values = self.value_head(last_hidden_states)[0].squeeze(-1)# (bs,)
        
        # token-level PPO: 使用sample-level RM的value_head的输出作为 last token 奖励
        # step-level & sample-level PPO：value_head取last token奖励
        return values # (bs,)
    
    def _forward_token_mix_rm(self,
                                     last_hidden_states: torch.Tensor, 
                                     attention_mask: torch.Tensor,
                                     action_mask: Optional[torch.Tensor]=None,) -> torch.Tensor:
        """token-mix-sample RM 推理

        Args:
            last_hidden_states (torch.Tensor):  LM 输出的 last hidden states
            action_mask (torch.Tensor):         action mask
            attention_mask (torch.Tensor):      attention mask

        Returns:
            torch.Tensor: _description_
        """
        
        # Token value head forward
        token_values = self.token_value_head(last_hidden_states)[0].squeeze(-1)
        
        # Value head forward，取last token value
        last_index =  torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in attention_mask], dtype=torch.int64)
        last_hidden_states = last_hidden_states[torch.arange(last_hidden_states.shape[0]), last_index]
        values = self.value_head(last_hidden_states)[0].squeeze(-1)# (bs,)
        
        if self.output_granularity == RewardModelGranularity.token_mix_sample:
            token_values.index_put_(indices=[torch.arange(token_values.shape[0]), last_index],values=values) #  (bs,len_seq)
            return token_values # (bs, seq_len)
        
        # token-level PPO：使用token value head输出作为token奖励，并将value head输出加至last token上
        if self.output_granularity == ActorGranularity.token:
            self.running_token_level_reward.update_batch(x_batch=token_values, value_mask=F.pad(action_mask, (1,0), value=False))
            self.running_sample_level_reward.update_batch(x_batch=values)
            self.running_token_level_reward.sync()
            self.running_sample_level_reward.sync()
            values = (values - self.running_sample_level_reward.mean) / self.running_sample_level_reward.std
            token_values = ((token_values - self.running_token_level_reward.mean) * F.pad(action_mask, (1,0), value=False)) / self.running_token_level_reward.std
            token_values.index_put_(indices=[torch.arange(token_values.shape[0]), last_index],values=values) #  (bs,len_seq)
            return token_values # (bs, seq_len)
        
        # sample-level step-level PPO：取token value head输出均值与value head输出相加，作为sample-level奖励
        if action_mask is not None:
            token_mean_values = masked_mean(token_values, F.pad(action_mask, (1,0), value=False), dim=-1).view(-1)
            values = values + token_mean_values
            
        return values # (bs,)

    def logging(self, step):
        """使用logger记录混合粒度奖励的分布

        Args:
            step (_type_): _description_
        """        
        if self.enable_reward_aligning and self.logger is not None:
            logging_rank_0(f"log reward aligning.", LoggingLevel.DEBUG)
            self.logger.log_metrics(
                {
                    "token_mean": self.running_token_level_reward.mean.item(),
                    "token_std": self.running_token_level_reward.std.item(),
                    "sample_mean": self.running_sample_level_reward.mean.item(),
                    "smaple_std": self.running_sample_level_reward.std.item(),
                }, step=step, metrics_group='reward_mixing')
        
        return

    def recover_reward(self, mean:torch.Tensor, std: torch.Tensor, is_token_level:bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """恢复混合奖励的分布

        Args:
            mean (torch.Tensor): _description_
            std (torch.Tensor): _description_
            is_token_level (bool, optional): _description_. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """       
        if self.enable_reward_aligning and self.logger is not None:
            if is_token_level:
                return mean * self.running_token_level_reward.std + self.running_token_level_reward.mean, self.running_token_level_reward.std
            else:
                return mean * self.running_sample_level_reward.std + self.running_sample_level_reward.mean, self.running_sample_level_reward.std
        
        return mean, std

class LlamaFSRewardModel(LlamaPreTrainedModel):
    def __init__(self, config:FengshenConfig):
        super().__init__(config)

        self.llama = FengshenLlamaModel(config)
        # self.value_head = torch.nn.Linear(self.config.hidden_size, 1, dtype=config.torch_dtype)
        self.value_head = mpu.RowParallelLinear(
            config=config,
            input_size=config.hidden_size,
            output_size=1,
            input_is_parallel=False,
            skip_bias_add=False,
            parallel_output=False,
        ) 
        # Initialize weights and apply final processing
        self.post_init()

class LlamaFSRewardModel_Mix(LlamaPreTrainedModel):
    def __init__(self, config:FengshenConfig):
        super().__init__(config)

        self.llama = FengshenLlamaModel(config)
        # self.value_head = torch.nn.Linear(self.config.hidden_size, 1, dtype=config.torch_dtype)
        self.value_head = mpu.RowParallelLinear(
            config=config,
            input_size=config.hidden_size,
            output_size=1,
            input_is_parallel=False,
            skip_bias_add=False,
            parallel_output=False,
        ) 
        # self.token_value_head = torch.nn.Linear(self.config.hidden_size, 1, dtype=config.torch_dtype)
        self.token_value_head = mpu.RowParallelLinear(
            config=config,
            input_size=config.hidden_size,
            output_size=1,
            input_is_parallel=False,
            skip_bias_add=False,
            parallel_output=False,
        )         

        # Initialize weights and apply final processing
        self.post_init()

def modeling_fengshenLlama_rm(pretrained_path:str,
                              lora_rank:int=0,
                              rm_granularity: RewardModelGranularity=None,
                              actor_granularity: ActorGranularity=None,
                              convert_func:Optional[Callable]=None,
                              logger:Optional[Logger]=None,
                              **kwargs) -> LlamaRM:
    """根据不同配置构造Reward model

    Args:
        pretrained_path (str):                              RM模型参数目录
        convert_func (Optional[Callable], optional):        文本转换方法，从Policy的文本格式转换至RM的文本格式. Defaults to None.
        lora_rank (int, optional):                          RM的LoRA rank. Defaults to 0.
        convertlogger_func (Optional[Callable], optional):  logger. Defaults to None.
        
        #### 以下为新版参数 ####
        
        rm_granularity (Optional[RewardModelGranularity], optional):    RM训练阶段采用的粒度. Defaults to None.
        actor_granularity (Optional[PPOGranularity], optional):         policy采用的粒度. Defaults to None.
        
        #### 以下为旧版参数 ####
        
        token_level_reward (Optional[bool], optional):  是否返回token粒度的奖励值. Defaults to None.
        step_level_reward (Optional[bool], optional):   step_level_reward（一句话说不清楚 [=_=] ）. Defaults to None.
        mix_reward (Optional[bool], optional):          是否将token粒度和sample粒度的奖励混合后输出（前提是RM类型为混合粒度，即mix）. Defaults to None.
        is_mix_rm (Optional[bool], optional):           RM是否为混合粒度，即具有value_head和token_value_head，分别输出于sample粒度奖励和token粒度奖励. Defaults to None.
        
        #######################
        
        NOTE: 不兼容旧版的参数，请按照说明使用新版的参数
        
    Returns:
        LlamaRM: _description_
    """    
    
    # 参数检查
    if not isinstance(rm_granularity, RewardModelGranularity):
        logging_rank_0(f"'modeling_fengshenLlama_rm' requires 'rm_granularity'.", LoggingLevel.ERROR)
        raise AttributeError
    elif not isinstance(actor_granularity, ActorGranularity):
        logging_rank_0(f"'modeling_fengshenLlama_rm' requires 'ppo_granularity'.", LoggingLevel.ERROR)
        raise AttributeError
    
    for key in kwargs:
        logging_rank_0(f"Deprecation: {key} has removed in 'modeling_fengshenLlama_rm', please use 'rm_granularity' and 'ppo_granularity'.", LoggingLevel.WARNING)
    
    # 实例化 RM
    CurrLlamaFSRewardModel = LlamaFSRewardModel_Mix if rm_granularity == RewardModelGranularity.token_mix_sample else LlamaFSRewardModel
    llama_model = CurrLlamaFSRewardModel.from_pretrained(pretrained_path)
    reward_model = LlamaRM(
        model=llama_model.llama,
        value_head=llama_model.value_head,
        token_value_head=llama_model.token_value_head if rm_granularity == RewardModelGranularity.token_mix_sample else None,
        rm_granularity=rm_granularity,
        output_granularity=actor_granularity,
        convert_func=convert_func,
        lora_rank=lora_rank,
        logger=logger,
    )
    return reward_model

class LlamaHFRewardModel(PreTrainedModel):
    # hf model for reward model
    # load like: model = LlamaHFRewardModel.from_pretrained(hf_rm_paht,granularity="sample").to(torch.bfloat16).cuda().eval()
    # granularity must be in ["sample","token","token_sample_mix"]

    config_class =LlamaConfig
    
    def __init__(self, config, granularity="sample"):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.value_head = torch.nn.Linear(config.hidden_size, 1) 
        self.token_value_head = torch.nn.Linear(config.hidden_size, 1)
        assert granularity in ["sample","token","token_sample_mix"], f"RM granularity must be in [\"sample\",\"token\",\"token_sample_mix\"], current RM granularity: {granularity}"
        self.granularity = granularity
    
    def forward(self,
                input_ids: torch.LongTensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(input_ids,attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        if self.granularity == "token":
            values = self.value_head(hidden_states).squeeze(-1)
            return values
        
        if attention_mask is None:
            last_hidden_states = hidden_states[:, -1]
        else:
            last_index =  torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in attention_mask], dtype=torch.int64)
            last_hidden_states = hidden_states[torch.arange(hidden_states.shape[0]), last_index]
        if self.granularity == "sample":
            values = self.value_head(last_hidden_states).squeeze(-1)
            return values
        if self.granularity == "token_sample_mix":
            token_values = self.token_value_head(hidden_states).squeeze(-1) #  (bs,len_seq)
            values = self.value_head(last_hidden_states).squeeze(-1) # (bs,)
            token_values.index_put_(indices=[torch.arange(token_values.shape[0]), last_index],values=values)
            return token_values

