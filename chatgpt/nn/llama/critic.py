# encoding=utf-8
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fengshen_inner.models.llama.configuration_llama import \
    LlamaConfig as FengshenConfig
from fengshen_inner.models.llama.modeling_llama import \
    LlamaModel as FengshenLlamaModel
from fengshen_inner.models.llama.modeling_llama import LlamaPreTrainedModel

from chatgpt.nn import Critic
from chatgpt.pipeline.config import ActorGranularity
from chatgpt.utils import CostTimer, LoggingLevel, logging_rank_0


class LlamaCritic(Critic):
    """
    GPT-NeoX Actor model.

    Args:
        model: Pretrained model
    """

    def __init__(self,
                 model,
                 ppo_granularity: ActorGranularity,
                 value_head: Optional[nn.Module]=None,) -> None:
        super().__init__(model, value_head)
        self.ppo_granularity = ppo_granularity
        return

    def forward(self,
                sequences: torch.LongTensor,
                action_mask: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Critic 推理，返回每个 state 的 value

        Args:
            sequences (torch.LongTensor):   token ids (bs, seq_len)
            action_mask (torch.Tensor):     sequences 中 action 的位置 (bs, seq_len-1)
            attention_mask (torch.Tensor):  sequences 中有效 token 的位置 (bs, seq_len)

        Returns:
            torch.Tensor: values
        """        
        
        with CostTimer():
            outputs = self.model(sequences, attention_mask=attention_mask, output_hidden_states=True, use_cache=True)
            last_hidden_states = outputs.hidden_states[-1]
            
            # token-level PPO: 返回每个 token 的 value
            if self.ppo_granularity is ActorGranularity.token:
                values = self.value_head(last_hidden_states)[0].squeeze(-1)[:, :-1]   # 这里要错位，取statede的隐层  
                return values # (bs, seq_len - 1)

            # sample-level & step-level PPO: 取 input 部分最后一个 token 的 value
            # 计算 input mask 并取 input last token idx
            input_mask = attention_mask ^ F.pad(action_mask, (1,0), value=False)
            last_index =  torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in input_mask], dtype=torch.int64)
            last_hidden_states = last_hidden_states[torch.arange(last_hidden_states.shape[0]), last_index]
            values = self.value_head(last_hidden_states)[0].squeeze(-1)
            
        return values # (bs,)

class LlamaFSCriticModel(LlamaPreTrainedModel):
    def __init__(self, config:FengshenConfig):
        super().__init__(config)

        self.llama = FengshenLlamaModel(config)
        self.value_head = torch.nn.Linear(self.config.hidden_size, 1, dtype=config.torch_dtype)
        # Initialize weights and apply final processing
        self.post_init()

def modeling_fengshenLlama_critic(pretrained_path:str,
                                  return_mean:Optional[bool]=None,
                                  ppo_granularity:Optional[ActorGranularity]=None) -> LlamaCritic:
    """构造 Llama Critic

    Args:
        pretrained_path (str):                                  Actor模型目录
        ppo_granularity (Optional[PPOGranularity], optional):   PPO训练粒度. Defaults to None.
        return_mean (Optional[bool], optional):                 是否返回平均后的 values（已弃用）. Defaults to None.

    Returns:
        LlamaCritic: Critic模型
    """    
    if not isinstance(ppo_granularity, ActorGranularity):
        logging_rank_0(f"'modeling_fengshenLlama_rm' requires 'ppo_granularity'.", LoggingLevel.ERROR)
        raise AttributeError
        
    if return_mean is not None:
        logging_rank_0(f"Deprecation: {return_mean} has removed in 'modeling_fengshenLlama_critic', please use 'ppo_granularity'.", LoggingLevel.WARNING)
    
    llama_model = LlamaFSCriticModel.from_pretrained(pretrained_path)
    critic_model = LlamaCritic(
        model=llama_model.llama,
        value_head=None,
        ppo_granularity=ppo_granularity
    )

    return critic_model