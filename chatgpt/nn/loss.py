from typing import Any, Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from fengshen_inner.models.megatron import mpu

from .utils import masked_mean
from chatgpt.utils import is_rank_0, print_rank_0, logging_rank_0
import gc


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    
def check_mp_eq(t:torch.Tensor):
    dp_world_size = mpu.get_model_parallel_world_size()
    synced_t = [torch.zeros_like(t, device=t.device) for _ in range(dp_world_size)]
    dist.all_gather(synced_t, t, group=mpu.get_model_parallel_group())
    
    eq = True
    for t1, t2 in zip(synced_t[:-1], synced_t[1:]):
        eq &= torch.all(t1.eq(t2)).item()
        
    return eq


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2, token_level_mean: bool = False) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.token_level_mean = token_level_mean

    def forward(self,
                log_probs: torch.Tensor,
                old_log_probs: torch.Tensor,
                advantages: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None,
                return_ratio: bool = False) -> torch.Tensor:
        ratio = (log_probs - old_log_probs).exp()
        # ratio = torch.where(torch.isinf(ratio), torch.zeros_like(ratio), ratio)
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        
        ## token级别均值，要求输入action_mask
        if self.token_level_mean and action_mask is not None:
            loss = torch.sum(loss * action_mask) / torch.sum(action_mask)
            ratio = torch.sum(ratio * action_mask) / torch.sum(action_mask)
        ## 样本级别均值
        else:
            if action_mask is not None:
                loss = masked_mean(loss, action_mask)
            loss = loss.mean()
            ratio = masked_mean(ratio, action_mask)
        
        return loss, ratio if return_ratio else loss


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.4, token_level_mean: bool = False) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.token_level_mean = token_level_mean

    def forward(self,
                values: torch.Tensor,
                old_values: torch.Tensor,
                reward: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
        surr1 = (values_clipped - reward)**2
        surr2 = (values - reward)**2
        loss = torch.max(surr1, surr2)
        if self.token_level_mean:
            loss = torch.sum(loss * action_mask) / torch.sum(action_mask)
        else:
            if action_mask is not None and loss.ndim > 1:
                loss = masked_mean(loss, action_mask, dim=-1)
            loss = loss.mean()
        return loss


class PPOPtxActorLoss(nn.Module):
    """
    To Do:

    PPO-ptx Actor Loss
    """

    def __init__(self, policy_clip_eps: float = 0.2, pretrain_coef: float = 0.0, pretrain_loss_fn=GPTLMLoss()) -> None:
        super().__init__()
        self.pretrain_coef = pretrain_coef
        self.policy_loss_fn = PolicyLoss(clip_eps=policy_clip_eps)
        self.pretrain_loss_fn = pretrain_loss_fn

    def forward(self,
                log_probs: torch.Tensor,
                old_log_probs: torch.Tensor,
                advantages: torch.Tensor,
                lm_logits: torch.Tensor,
                lm_input_ids: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        policy_loss = self.policy_loss_fn(log_probs, old_log_probs, advantages, action_mask=action_mask)
        lm_loss = self.pretrain_loss_fn(lm_logits, lm_input_ids)
        return policy_loss + self.pretrain_coef * lm_loss


class EntropyLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        return
    
    def forward(self,
                logits:torch.Tensor,
                action_mask:torch.Tensor):
        
        logprobs = F.log_softmax(logits, dim=-1)                                # (bs, seq_len, vocab_size)
        token_entropy = -1 * torch.sum(logprobs * torch.exp(logprobs), dim=-1)  # (bs, seq_len)
        masked_token_entropy = token_entropy * action_mask                      # (bs, seq_len)
        sample_entropy = torch.mean(masked_token_entropy, dim=-1)               # (bs,)
        entropy = torch.mean(sample_entropy)                                    # (1,)
        return entropy

class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(chosen_reward - reject_reward)
        log_probs = torch.log(probs)
        loss = -log_probs.mean()
        return loss

class PreferenceLoss(nn.Module):
    """
    Preference Loss for EDPO
    """
    def __init__(self, beta: float = 0.5) -> None:
        super().__init__()
        self.beta = beta

    def forward(self,
                log_probs: torch.Tensor,# prompt_bs * preference_bs * seq_length
                ref_log_probs: Optional[torch.Tensor]=None,
                action_mask: Optional[torch.Tensor]=None,
                preference_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        if action_mask is None:
            action_mask = torch.ones_like(log_probs)
        if ref_log_probs is None:
            ref_log_probs = torch.zeros_like(log_probs)
        if preference_mask is None:
            preference_mask = torch.ones_like(exp_log_ratios)
        torch.set_printoptions(profile="full")

        # action_lens = torch.sum(action_mask, dim=-1) # prompt_bs * preference_bs 
        # print_rank_0(f'action_lens {action_lens}, action_mask {action_mask}')
        # print_rank_0(f'log_probs {log_probs}, ref_log_probs {ref_log_probs}')
        log_ratio = ((log_probs - ref_log_probs)*action_mask).sum(dim=-1) # prompt_bs * preference_bs
        log_ratio *= preference_mask*self.beta
        # log_ratio *= preference_mask*self.beta/(action_lens + 1.0e-20)
        exp_log_ratios = log_ratio.exp()
        # print_rank_0(f'log_ratio {log_ratio}, exp_log_ratios {exp_log_ratios}')


        masked_exp_log_ratios = exp_log_ratios * preference_mask
        # print_rank_0(f'masked_exp_log_ratios {masked_exp_log_ratios}, preference_mask {preference_mask}')

        r2l_cumsum_exp_log_ratios = torch.flip(torch.flip(masked_exp_log_ratios, [1]).cumsum(dim=1), [1])
        # print_rank_0(f'r2l_cumsum_exp_log_ratios {r2l_cumsum_exp_log_ratios}')
        
        loss = masked_mean((exp_log_ratios/(r2l_cumsum_exp_log_ratios+1.0)).log(), preference_mask) # prompt_bs

        loss_sample = -loss.clone().detach()
        # print_rank_0('loss', loss)
        loss = -loss.mean() 

        reward = log_ratio.clone().detach().mean(dim=-1)

        # print_rank_0('loss_mean', loss)

        torch.cuda.empty_cache()
        gc.collect()

        return loss, reward, loss_sample


class GeneralizedKDLoss(nn.Module):

    support_divergence = ['KLD', 'RKLD', 'JSD', 'TVD']

    def __init__(self, 
                 divergence: str,
                 JSD_coef: float = 0.5,
                 temperature: float = 1.0,
                 level: str = 'token',
                 eps: float = 0,
                 pseudo_JSD: bool = True,
                 use_CE: bool = False,
                 fdistill: bool = False) -> None:
        super().__init__()

        divergence = divergence.upper().strip()
        assert divergence in self.support_divergence, \
            f"Got divergence type {divergence} but only support {self.support_divergence}"
        self.divergence = divergence

        self.JSD_coef = JSD_coef
        self.eps = eps
        self.temperature = temperature
        self.level = level

        self.pseudo_JSD = pseudo_JSD
        self.use_CE = use_CE
        self.fdistill = fdistill
        if not self.pseudo_JSD or self.use_CE or self.fdistill:
            raise NotImplementedError("'pseudo_JSD', 'use_CE', and 'fdistill' are future features")

        if not self.use_CE:
            if self.level == 'token':
                self.KL_loss_func = nn.KLDivLoss(reduction="batchmean", log_target=True)
            elif self.level == 'sequence':
                self.KL_loss_func = nn.KLDivLoss(reduction="sum", log_target=True)
            else:
                raise NotImplementedError(f"Supported level: 'token', 'sequence', but got {self.level}")

    def forward(self, 
                src_logits: torch.Tensor, 
                trg_logits: torch.Tensor, 
                src_action_mask: torch.Tensor, 
                trg_action_mask: torch.Tensor) -> torch.Tensor:

        src_logits, src_action_mask = self._dim_check(src_logits, src_action_mask)
        trg_logits, trg_action_mask = self._dim_check(trg_logits, trg_action_mask)

        if self.divergence == 'KLD':
            return self._forward_KLD(src_logits, trg_logits, src_action_mask, trg_action_mask)
        elif self.divergence == 'RKLD':
            return self._forward_RKLD(src_logits, trg_logits, src_action_mask, trg_action_mask)
        elif self.divergence == 'JSD':
            return self._forward_JSD(src_logits, trg_logits, src_action_mask, trg_action_mask)
        else:
            return self._forward_TVD(src_logits, trg_logits, src_action_mask, trg_action_mask)

    def _forward_KLD(self, 
                src_logits: torch.Tensor, 
                trg_logits: torch.Tensor, 
                src_action_mask: torch.Tensor, 
                trg_action_mask: torch.Tensor) -> torch.Tensor:
        src_action_log_probs = self._build_probs(src_logits, src_action_mask)
        trg_actoin_log_probs = self._build_probs(trg_logits, trg_action_mask)
        if self.level =='token':
            return self.KL_loss_func(src_action_log_probs, trg_actoin_log_probs)
        if self.level == 'sequence':
            return self.KL_loss_func(src_action_log_probs, trg_actoin_log_probs) / src_logits.shape[0]
    
    def _forward_RKLD(self, 
                src_logits: torch.Tensor, 
                trg_logits: torch.Tensor, 
                src_action_mask: torch.Tensor, 
                trg_action_mask: torch.Tensor) -> torch.Tensor:
        src_action_log_probs = self._build_probs(src_logits, src_action_mask)
        trg_actoin_log_probs = self._build_probs(trg_logits, trg_action_mask)
        if self.level == 'token':
            return self.KL_loss_func(trg_actoin_log_probs, src_action_log_probs)
        if self.level == 'sequence':
            return self.KL_loss_func(trg_actoin_log_probs, src_action_log_probs) / src_logits.shape[0]

    def _forward_JSD(self, 
                src_logits: torch.Tensor, 
                trg_logits: torch.Tensor, 
                src_action_mask: torch.Tensor, 
                trg_action_mask: torch.Tensor) -> torch.Tensor:
        src_action_probs = self._build_probs(src_logits, src_action_mask, return_log=False)
        trg_actoin_probs = self._build_probs(trg_logits, trg_action_mask, return_log=False)
        interpolation = self.JSD_coef * src_action_probs + (1 - self.JSD_coef) * trg_actoin_probs

        src_action_probs = src_action_probs.log()
        trg_actoin_probs = trg_actoin_probs.log()
        interpolation = interpolation.log()
        loss = self.JSD_coef * self.KL_loss_func(interpolation, src_action_probs) + \
                (1 - self.JSD_coef) * self.KL_loss_func(interpolation, trg_actoin_probs)
        if self.level == 'token':
            return loss
        if self.level == 'sequence':
            return loss / src_logits.shape[0]
            
    def _forward_TVD(self, 
                src_logits: torch.Tensor, 
                trg_logits: torch.Tensor, 
                src_action_mask: torch.Tensor, 
                trg_action_mask: torch.Tensor) -> torch.Tensor:
        src_action_probs = self._build_probs(src_logits, src_action_mask, return_log=False)
        trg_actoin_probs = self._build_probs(trg_logits, trg_action_mask, return_log=False)
        if self.level == 'token':
            return torch.abs(src_action_probs - trg_actoin_probs).sum() / (2 * src_action_probs.shape[0])
        if self.level == 'sequence':
            return torch.abs(src_action_probs - trg_actoin_probs).sum() / (2 * src_logits.shape[0])
    
    def _build_probs(self, logits: torch.Tensor, action_mask: torch.Tensor, return_log: bool = True) -> torch.Tensor: 
        vocab_size = logits.shape[-1]
        action_logits = torch.masked_select(logits, action_mask)
        action_logits = self.eps + action_logits.view(-1, vocab_size) / self.temperature
        if return_log:
            return F.log_softmax(action_logits, dim=-1)
        else:
            return F.softmax(action_logits, dim=-1)
        
    def _dim_check(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert logits.dim() == 3, f"Expect logits of dim 3 but got dim {logits.dim()}"
        if mask.dim() == 2:
            assert logits.shape[: 2] == mask.shape, \
                f"logits shape {logits.shape}, action shape {mask.shape}"                
            mask = mask[:, :, None].expand_as(logits)
        elif mask.dim() == 3:
            assert logits.shape == mask.shape, \
                f"logits shape {logits.shape}, action shape {mask.shape}"
        else:
            raise ValueError('Action mask should be 2-dimensional or 3-dimensional.')
        return logits, mask
        

class EOSLoss(nn.Module):
    def __init__(self, eos_token_id: int):
        super().__init__()
        self.eos_token_id = eos_token_id

    def forward(self, logprob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # eos_pos = torch.cumsum(mask, dim=-1).max(dim=-1).indices
        # logprob = F.log_softmax(logits, dim=-1)
        # logprob = logprob[:,:, self.eos_token_id]
        # logprob = logprob.gather(dim=1, index=eos_pos.unsqueeze(1))   
        # return logprob.mean()
        eos_pos = torch.cumsum(mask, dim=-1).max(dim=-1).indices
        loss = logprob.gather(dim=1, index=eos_pos.unsqueeze(1))
        return loss.mean()   