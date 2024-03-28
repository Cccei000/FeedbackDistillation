# enoding=utf-8
import copy
import gc
import heapq
import re
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM
from fengshen_inner.models.llama.modeling_llama_lora import \
    LlamaForCausalLMLora

from chatgpt.nn import Actor, GSArgs, TotGS, RewardModel
from chatgpt.nn.utils import log_probs_from_logits, zero_pad_sequences
from chatgpt.pipeline.config import ActorGranularity, RewardModelGranularity
from chatgpt.utils import CostTimer, LoggingLevel, logging_rank_0


class LlamaActor(Actor):
    """
    Llama Actor model.

    Args:
        model: Pretrained model
    """

    def __init__(self,
                 model:Union[LlamaForCausalLM, LlamaForCausalLMLora],
                 actor_granularity:ActorGranularity,
                 rm_granularity:RewardModelGranularity,) -> None:
        super().__init__(model)
        self.actor_granularity=actor_granularity
        self.rm_granularity=rm_granularity
        return
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        gs_args: Optional[GSArgs] = None,
        reward_model: Optional[RewardModel] = None,
        for_validation: bool = False,
        **kwargs):
        """Actor模型生成方法

        Args:
            input_ids (torch.Tensor): _description_
            gs_args (Optional[GSArgs], optional): _description_. Defaults to None.
            reward_model (Optional[RewardModel], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """        
        
        pad_token_id = kwargs.get('pad_token_id', None)
        eos_token_id = kwargs.get('eos_token_id', None)

        ## Attention! GenerationMixin.sample() has been rewriten by nn.generation._sample
        # 
        # print_rank_0(f"Rank-{self.model.device}: START SAMPLE!")
        
        # 确定生成方法
        
       
        if gs_args is None:
            # logging_rank_0(f"Rank-{self.model.device}: do vallina search", level='debug')
            with CostTimer():
                output = self.model.generate(input_ids.to(self.model.device), **kwargs)
        elif gs_args.enabling_tot:
            # logging_rank_0(f"Rank-{self.model.device}: do bfs search", level='debug')
            output = self.tot_search(input_ids.to(self.model.device), reward_model, gs_args, **kwargs)
        elif gs_args.enabling_bon:
            # logging_rank_0(f"Rank-{self.model.device}: do best-of-n search", level='debug')
            output = self.bestofn_generate(input_ids.to(self.model.device), reward_model, gs_args, **kwargs)
        else:
            with CostTimer():
                output = self.model.generate(input_ids.to(self.model.device), **kwargs)        
        
        # NOTE:
        #   vallina search 会返回细粒度的生成信息 gen_info
        #   这里是为了对齐不同search方法的输出
        if isinstance(output, tuple):
            sequences, gen_info = output
        else:
            sequences, gen_info = output, []
            
        logging_rank_0(f"Finish sample, use time {CostTimer.get_time()} | {sequences.shape}", level=LoggingLevel)

        torch.set_printoptions(edgeitems=1000, linewidth=1000)

        if sequences.shape[1] < input_ids.shape[1]:
            sequences = F.pad(sequences, (0, input_ids.shape[1] - sequences.shape[1]), value=eos_token_id)
            
        # # 不同粒度的PPO处理生成结果
        if self.actor_granularity is ActorGranularity.token:
            return self._process_token_ppo_output(
                input_ids=input_ids,
                sequences=sequences,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id
            ) + (gen_info,)
        elif self.actor_granularity is ActorGranularity.sample:
            return self._process_sample_ppo_output(
                input_ids=input_ids,
                sequences=sequences,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id
            ) + (gen_info,)
        # Sample-level PPO
        else:
            # 验证
            if for_validation:
                return self._process_sample_ppo_output(
                    input_ids=input_ids,
                    sequences=sequences,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id
                )
            # 训练
            else:
                return self._process_step_ppo_output(
                    input_ids=input_ids,
                    sequences=sequences,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    gs_args=gs_args,
                    reward_model=reward_model
                )
        
        
    def _process_step_ppo_output(self,
                                 input_ids: torch.Tensor,
                                 sequences: torch.Tensor,
                                 pad_token_id: int,
                                 eos_token_id: int,
                                 gs_args: GSArgs,
                                 reward_model: RewardModel,):
        """step level PPO处理Actor的生成结构

        Args:
            input_ids (torch.Tensor):   输入token ids
            sequences (torch.Tensor):   输入token ids + 模型生成token ids
            pad_token_id (int):         pad token id
            eos_token_id (int):         eos token id
            gs_args (GSArgs):           search generation 的配置
            reward_model (RewardModel): 奖励模型实例

        Returns:
            _type_: _description_
        """        
        
        #### 以下代码将完整文本切分为多步样本的代码 ####
        fg_sequence = []
        fg_input_mask = []
        fg_rewards = []
        step_reward = []
        step_id = []
        # sequence decode后拆分
        for sequence_i, input_ids_i in zip(sequences,input_ids):
            sequences_i = []
            inputs_ids_i_list = []
            input_ids_prev = input_ids_i    # 标记初始input_ids
            # input 实际长度判断有效步骤
            input_i_len_index = torch.nonzero((input_ids_i == pad_token_id).cumsum(dim=-1) == 1).squeeze().reshape(-1).tolist()
            input_i_len = input_i_len_index[0] if len(input_i_len_index)>0 else len(input_ids_i)
            # 按分隔符拆分response
            out_text = gs_args.generator_tokenizer.decode(sequence_i.detach().tolist(), skip_special_tokens=False).strip().replace('<s>','').replace('</s>','')
            # out_text_splits = out_text.split('\n\n')
            # TODO：多分隔符
            out_text_splits = re.split(r'\n', out_text)
            # 分步计算reward
            for step_index in range(len(out_text_splits)):
                # TODO：单步预测 是否预测分隔符，多分隔符记录前一步text
                # curr_step_text = out_text[:len(out_text)-len(splitted[-1])]
                curr_step_text = '\n'.join(out_text_splits[:step_index+1])
                # curr_step_text = '\n\n'.join(out_text_splits[:step_index+1]) + '\n\n'
                curr_step_ids = gs_args.generator_tokenizer(curr_step_text).input_ids
                # 保证分步只在response拆分
                if len(curr_step_ids) <= input_i_len:
                    continue
                # 单步太短时，合并多步
                if (step_index < len(out_text_splits)-1) and len(curr_step_ids)-input_ids_prev.shape[0] < gs_args.min_step_lengths:
                    continue
                curr_step_ids.append(eos_token_id)
                curr_step_ids_tensor = torch.tensor(curr_step_ids)
                sequences_i.append(curr_step_ids_tensor)  #当前样本分步sequences
                fg_sequence.append(curr_step_ids_tensor)
                inputs_ids_i_list.append(input_ids_prev)
                fg_input_mask.append(input_ids_prev)
                input_ids_prev =  curr_step_ids_tensor    # 标记上一轮input_ids
            sequences_i_tensor = zero_pad_sequences(sequences_i, side="right", padding_value=pad_token_id)
            sequences_i_tensor = sequences_i_tensor.to(self.model.device)   #(n_i, seq_len)
            #input mask
            input_mask_i_tensor = zero_pad_sequences(inputs_ids_i_list, side="right", padding_value=pad_token_id)
            input_mask_i_tensor = (input_mask_i_tensor == pad_token_id).cumsum(dim=-1) != 0 # 标记input_ids中的pad token
            input_mask_i_tensor = input_mask_i_tensor.to(self.model.device)

            if sequences_i_tensor.shape[1] < input_mask_i_tensor.shape[1]:
                sequences_i_tensor = F.pad(sequences_i_tensor, (0, input_mask_i_tensor.shape[1] - sequences_i_tensor.shape[1]), value=eos_token_id)

            # attention mask 
            attention_mask_i = None
            if pad_token_id is not None:
                if eos_token_id is not None and eos_token_id == pad_token_id:
                    attention_mask_i = (sequences_i_tensor == pad_token_id).cumsum(dim=-1) <= 1
                else:
                    attention_mask_i = sequences_i_tensor.not_equal(pad_token_id)
                attention_mask_i.to(dtype=torch.long, device=sequences.device)
            
            # action mask
            if eos_token_id is None:
                action_mask_i = torch.ones_like(sequences_i_tensor, dtype=torch.bool)          # (bs, seq_len)
            else:
                action_mask_i = (sequences_i_tensor == eos_token_id).cumsum(dim=-1) == 0       # (bs, seq_len)
                action_mask_i = F.pad(action_mask_i, (1, -1), value=True)               # (bs, seq_len)
            action_mask_i[:, :input_mask_i_tensor.shape[1]] *= input_mask_i_tensor
            action_mask_i = action_mask_i[:, 1:]
            action_mask_i = action_mask_i.to(sequences.device)

            # 计算分步reward 
            # 记录每条样本的step_reward
            fg_rewards_i = []
            bs_rm = gs_args.gs_eval_batch_size
            for seq_index in range(0, sequences_i_tensor.shape[0], bs_rm):
                start, end = seq_index, seq_index + bs_rm
                sequences_bs = sequences_i_tensor[start:end]
                attention_mask_bs = attention_mask_i[start:end]
                action_mask_bs = action_mask_i[start:end]
                scores = reward_model(sequences_bs, action_mask_bs, attention_mask_bs)
                fg_rewards_i.extend(scores.detach().tolist())

            if self.rm_granularity is not RewardModelGranularity.token:
                # 计算分步reward delta
                # 计算Fine-grained Reward Delta
                fg_rewards_i = [(fg_rewards_i[ri] if ri==0 else fg_rewards_i[ri]-fg_rewards_i[ri-1]) for ri in range(len(fg_rewards_i))]
                # 写作任务step0奖励值异常，尝试手动set为reward_step_0 = 0
                # fg_rewards_i = [(0.0 if ri==0 else fg_rewards_i[ri]-fg_rewards_i[ri-1]) for ri in range(len(fg_rewards_i))]

            fg_rewards.extend(fg_rewards_i)  #(num, )
            # 返回样本i的step_reward，GAE计算
            step_len = len(fg_rewards_i)
            for step_index in range(step_len):
                step_reward.append(torch.tensor(fg_rewards_i))  #(num, steps)
                step_id.append(step_len-step_index)
        
        step_reward = zero_pad_sequences(step_reward, side="left", padding_value=0) #(n, steps) # TODO: pad_id ???
        sequences = zero_pad_sequences(fg_sequence, side="right", padding_value=pad_token_id)   #(n, seq_len)
        input_mask_all = zero_pad_sequences(fg_input_mask, side="right", padding_value=pad_token_id)
        input_mask = (input_mask_all == pad_token_id).cumsum(dim=-1) != 0    #标记input_ids中的pad token
        # input_mask = torch.cat(fg_input_mask, dim=0)
        sequences = sequences.to(self.model.device)   #(n, seq_len)
        input_mask = input_mask.to(self.model.device)   #(n, seq_len)
        step_reward = step_reward.to(self.model.device) #(n, steps)
        step_id = torch.tensor(step_id).to(device=self.model.device) #(n,)
        
        #### 以上代码将完整文本切分为多步样本的代码 ####
        
        if sequences.shape[1] < input_ids.shape[1]:
            sequences = F.pad(sequences, (0, input_ids.shape[1] - sequences.shape[1]), value=eos_token_id)
        attention_mask = self._create_attn_mask(sequences, eos_token_id, pad_token_id)
        action_mask = self._create_action_mask(input_mask, sequences, eos_token_id, pad_token_id)
        fg_rewards = torch.tensor(fg_rewards)
        fg_rewards.to(self.model.device)
        
        return sequences, attention_mask, action_mask[:, 1:], fg_rewards, step_reward, step_id  # (num, seq_len), (num, seq_len), (num, seq_len - 1), (num,)
    
    def _process_token_ppo_output(self, **kwargs):
        """token level PPO 处理Actor生成结果，同 sample level PPO

        Returns:
            _type_: _description_
        """        
        # 和sample level共用代码
        return self._process_sample_ppo_output(**kwargs)
    
    def _process_sample_ppo_output(self,
                                   input_ids: torch.Tensor,
                                   sequences: torch.Tensor,
                                   pad_token_id: int,
                                   eos_token_id: int,):
        """sample level PPO 处理Actor生成结果

        Args:
            input_ids (torch.Tensor):   输入token ids
            sequences (torch.Tensor):   输入token ids + 模型生成token ids
            pad_token_id (int):         pad token id
            eos_token_id (int):         eos token id
        
        Returns:
            _type_: _description_
        """        
        input_mask = (input_ids == pad_token_id).cumsum(dim=-1) != 0 # 标记input_ids中的pad token
        input_mask = input_mask.to(self.model.device)
        attention_mask = self._create_attn_mask(sequences, eos_token_id, pad_token_id)
        action_mask = self._create_action_mask(input_ids, sequences, eos_token_id, pad_token_id)
        return sequences, attention_mask, action_mask[:, 1:] # (bs, seq_len), (bs, seq_len), (bs, seq_len - 1)

    def _create_action_mask(self, input_ids:torch.Tensor, sequences:torch.Tensor, eos_token_id: Optional[int] = None, pad_token_id: Optional[int] = None) -> torch.Tensor:
        """构造 Action mask，指出 sequences 中 action

        Args:
            input_ids (torch.Tensor):                   输入token ids
            sequences (torch.Tensor):                   输入token ids + 模型生成token ids
            eos_token_id (Optional[int], optional):     eos token id. Defaults to None.
            pad_token_id (Optional[int], optional):     pad token id. Defaults to None.

        Returns:
            torch.Tensor: Action mask
        """              

        input_mask = (input_ids == pad_token_id).cumsum(dim=-1) != 0 # 标记input_ids中的pad token
        input_mask = input_mask.to(sequences.device)


        if eos_token_id is None:
            action_mask = torch.ones_like(sequences, dtype=torch.bool)          # (bs, seq_len)
        else:
            action_mask = (sequences == eos_token_id).cumsum(dim=-1) == 0       # (bs, seq_len)
            action_mask = F.pad(action_mask, (1, -1), value=True)               # (bs, seq_len)
        action_mask[:, :input_mask.shape[1]] *= input_mask
        return action_mask

    def _create_attn_mask(self, sequences:torch.Tensor, eos_token_id: Optional[int] = None, pad_token_id: Optional[int] = None) -> torch.Tensor:
        """构造 Attention mask，指出 sequences 中的有效 token

        Args:
            sequences (torch.Tensor): _description_
            eos_token_id (Optional[int], optional): _description_. Defaults to None.
            pad_token_id (Optional[int], optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: Attention mask
        """        
        attention_mask = None
        if pad_token_id is not None:
            if eos_token_id is not None and eos_token_id == pad_token_id:
                attention_mask = (sequences == pad_token_id).cumsum(dim=-1) <= 1
            else:
                attention_mask = sequences.not_equal(pad_token_id)
            attention_mask.to(dtype=torch.long, device=sequences.device)
        return attention_mask

    def forward(self,
                sequences: torch.LongTensor,
                num_actions: int,
                attention_mask: Optional[torch.Tensor] = None,
                return_logits: bool = False) -> torch.Tensor:
        """Actor 推理，返回 sequences 中各 token 的 logprob 和 logits

        Args:
            sequences (torch.LongTensor): _description_
            num_actions (int): _description_
            attention_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
            return_logits (bool, optional): _description_. Defaults to False.

        Returns:
            torch.Tensor: _description_
        """        
        # attention_mask = attention_mask[:, None, None, :]
        # 计算Actor forward时间
        with CostTimer():
            output = self.model(
                sequences, 
                attention_mask=attention_mask,
                use_cache=True,
            )
        logits = output[0]  # (bs, seq_len, vocab_size) 但是错位
        log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])  # (bs, seq_len - 1)

        if return_logits:
            return log_probs, logits[:, :-1, :] # (bs, seq_len - 1), (bs, seq_len - 1, vocab_size)

        return log_probs # (bs, seq_len - 1)

    def tot_search(self, input_ids, reward_model:nn.Module, tot_args:GSArgs, **kwargs):
        generate_kwargs = copy.deepcopy(kwargs)

        explorer = TotGS(self.model, reward_model, tot_args.generator_tokenizer, tot_args.evaluator_tokenizer, tot_args.gs_gen_batch_size, tot_args.gs_eval_batch_size, **generate_kwargs)
        result,_ = explorer.search(input_ids, T=tot_args.gs_iterations, k=tot_args.gs_gen_repeat_times, b=tot_args.gs_breadth)
        torch.cuda.empty_cache()
        gc.collect()
        return result
    
    def bestofn_generate(self, input_ids, reward_model:nn.Module, args:GSArgs, **kwargs):
        generate_kwargs = copy.deepcopy(kwargs)
        pad_token_id = kwargs.get('pad_token_id', None)
        eos_token_id = kwargs.get('eos_token_id', None)
        # TODO：按batch处理
        # Bset of N Generator
        reward_model = reward_model.cuda()
        rm_device = next(reward_model.parameters()).device
        top_n = args.best_of_n_times
        sequences = []
        rm_bs = args.gs_eval_batch_size
        gen_bs = args.gs_gen_batch_size
        for input_ids_i in input_ids:
            input_ids_i_n = input_ids_i.unsqueeze(0).repeat(top_n,1)
            sequences_bs = []
            for i in range(0, len(input_ids_i_n), gen_bs):
                input_ids_bs = input_ids_i_n[i:(i+gen_bs)]
                sequences_bs.extend(self.model.generate(input_ids_bs.to(self.model.device), **generate_kwargs)[0])
                
            sequences_i_tensor = zero_pad_sequences(sequences_bs, side="right", padding_value=pad_token_id)
            sequences_i_tensor = sequences_i_tensor.to(rm_device)   #(n_i, seq_len)
            #input mask
            input_mask_i_tensor = zero_pad_sequences(input_ids_i_n, side="right", padding_value=pad_token_id)
            input_mask_i_tensor = (input_mask_i_tensor == pad_token_id).cumsum(dim=-1) != 0 # 标记input_ids中的pad token
            input_mask_i_tensor = input_mask_i_tensor.to(rm_device)

            if sequences_i_tensor.shape[1] < input_mask_i_tensor.shape[1]:
                sequences_i_tensor = F.pad(sequences_i_tensor, (0, input_mask_i_tensor.shape[1] - sequences_i_tensor.shape[1]), value=eos_token_id)

            # attention mask 
            attention_mask_i = None
            if pad_token_id is not None:
                if eos_token_id is not None and eos_token_id == pad_token_id:
                    attention_mask_i = (sequences_i_tensor == pad_token_id).cumsum(dim=-1) <= 1
                else:
                    attention_mask_i = sequences_i_tensor.not_equal(pad_token_id)
                attention_mask_i.to(dtype=torch.long, device=sequences_i_tensor.device)
            
            # action mask
            if eos_token_id is None:
                action_mask_i = torch.ones_like(sequences_i_tensor, dtype=torch.bool)          # (bs, seq_len)
            else:
                action_mask_i = (sequences_i_tensor == eos_token_id).cumsum(dim=-1) == 0       # (bs, seq_len)
                action_mask_i = F.pad(action_mask_i, (1, -1), value=True)               # (bs, seq_len)
            action_mask_i[:, :input_mask_i_tensor.shape[1]] *= input_mask_i_tensor
            action_mask_i = action_mask_i[:, 1:]
            action_mask_i = action_mask_i.to(sequences_i_tensor.device)

            
            cand_scores = []
            for i in range(0, len(sequences_bs), rm_bs):
                sequences_rm_bs = sequences_i_tensor[i:i+rm_bs]
                attention_mask_bs = attention_mask_i[i:i+rm_bs]
                action_mask_bs = action_mask_i[i:i+rm_bs]
                # print(next(reward_model.parameters()).device, sequences_rm_bs.device, action_mask_bs.device, attention_mask_bs.device)
                scores = reward_model(sequences_rm_bs, action_mask_bs, attention_mask_bs)
                cand_scores.extend(scores.detach().tolist())
            top1 = heapq.nlargest(1, range(len(cand_scores)), key=lambda i: cand_scores[i])[0]
            sequences.append(sequences_bs[top1])
        sequences = zero_pad_sequences(sequences, side="right", padding_value=pad_token_id)
        sequences = sequences.to(rm_device)
        # reward_model = reward_model.cpu()
        return sequences
        
