from typing import Optional, Union, Tuple, List

import loralib as lora
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from fengshen_inner.models.megatron import mpu

from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence



def compute_approx_kl(log_probs: torch.Tensor,
                      log_probs_base: torch.Tensor,
                      return_mean: bool = True,
                      action_mask: Optional[torch.Tensor] = None, dim=1) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    approx_kl = (log_ratio.exp() - 1) - log_ratio
    if action_mask is not None and return_mean:
        return masked_mean(approx_kl, action_mask, dim=dim)
    elif action_mask is not None and not return_mean:
        return torch.where(action_mask, approx_kl, torch.zeros_like(approx_kl))
    
    return approx_kl.mean(dim=dim) if return_mean else approx_kl


def compute_reward(r: Union[torch.Tensor, float],
                   kl_coef: float,
                   log_probs: torch.Tensor,
                   log_probs_base: torch.Tensor,
                   action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if kl_coef <= 0.0:
        return r
    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
    reward = r - kl_coef * kl
    return reward


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    # 去除inf值
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = tensor / (mask_sum + 1e-8)
    return mean


def get_reward_by_mask(token_reward: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    last_index = torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in mask], device=token_reward.device, dtype=torch.long)
    reward = token_reward[torch.arange(0, token_reward.shape[0], device=token_reward.device), last_index]
    return reward


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


def normalize(tensor: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    mean = tensor.mean(dim)
    mean_centered = tensor - mean
    var = (mean_centered**2).mean(dim)
    norm = mean_centered * var.clamp(min=eps).rsqrt()
    return norm


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = 'right', padding_value: int = 0) -> torch.Tensor:
    assert side in ('left', 'right')
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == 'left' else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=padding_value))
    return torch.cat(padded_sequences, dim=0)


def shrink_mask(mask: torch.Tensor, shrink: Tuple) -> torch.Tensor:
    assert isinstance(shrink, tuple) and len(shrink) == 2
    lshrink, rshrink = shrink
    assert lshrink >= 0 and rshrink >= 0
    if not lshrink and not rshrink:
        return mask
    cumsum = torch.cumsum(mask, dim=-1)
    if lshrink:
        mask = torch.where(cumsum > lshrink, mask, False)
    if rshrink:
        mask = torch.where(cumsum <= cumsum.max(-1, keepdim=True).values - rshrink, mask, False)
    # eos_idx = torch.cumsum(mask, dim=-1).max(dim=-1).indices
    # mask[torch.arange(0, len(eos_idx)), eos_idx] = False
    return mask


def pad_2d_tensors(tensors: List[torch.Tensor], padding_value: int=0):
    max_rows = max([t.shape[0] for t in tensors])
    max_cols = max([t.shape[1] for t in tensors])

    padded_tensors = torch.full((len(tensors), max_rows, max_cols), padding_value, dtype=tensors[0].dtype)

    for i, t in enumerate(tensors):
        padded_tensors[i, :t.shape[0], :t.shape[1]] = t

    return padded_tensors

def pad_3d_tensors(tensor_list: List[torch.Tensor], padding_value: int = 0):
    # tensor_list = [torch.rand(2,5), torch.rand(3,4), torch.rand(1, 6)]
    max_size_row = max([t.size(0) for t in tensor_list])
    max_size_col = max([t.size(1) for t in tensor_list])

    padded_tensors = []
    mask_3d = []
    mask_2d = []

    for t in tensor_list:
        # 计算每个 tensor 在每个维度上需要填充的大小
        padding = (0, max_size_col - t.size(1), 0, max_size_row - t.size(0))  # (left, right, top, bottom) padding
        padded_tensors.append(pad(t, padding, value=padding_value))

        # 创建3D mask
        mask_3d.append(pad(torch.ones_like(t), padding, value=0))

        # 创建2D mask
        mask_2d.append(pad(torch.ones(t.size(0), 1), (0, 0, 0, max_size_row - t.size(0)), value=0))

    # 将填充后的 tensor 和 mask 堆叠成一个 3D tensor
    result = torch.stack(padded_tensors)
    mask_3d = torch.stack(mask_3d)
    mask_2d = torch.stack(mask_2d).squeeze(-1)  # 移除额外的维度
    
    return result, mask_3d, mask_2d

def concat_2d_tensors(tensors: List[torch.Tensor], padding_value: int=0):
    assert tensors[0].device == tensors[1].device
    tensor1_no_pad = [row[row != padding_value] for row in tensors[0]]

    concatenated_tensor = pad_sequence([torch.cat([row, tensors[1][i]]) for i,row in enumerate(tensor1_no_pad)], batch_first=True, padding_value=padding_value)

    return concatenated_tensor


def get_global_statistics(xs: torch.Tensor) -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes
    """
    sum_and_count = torch.tensor([xs.sum(), xs.numel()], device=xs.device)
    dist.all_reduce(sum_and_count, dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum((xs - global_mean) ** 2)
    dist.all_reduce(sum_var, dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
    global_var = sum_var / count
    return global_mean, global_var, count



def normalize_dist(xs: torch.Tensor, shift_mean=True, distributed=True, eps: float = 1e-8) -> torch.Tensor:
    """normalize dist tensor"""
    if distributed and dist.is_initialized():
        mean, var, _ = get_global_statistics(xs)
    else:
        var, mean = torch.var_mean(xs)

    whitened = (xs - mean) * torch.rsqrt(var + eps)
    if not shift_mean:
        whitened += mean
    return whitened


def convert_to_lora(model: nn.Module,
                    input_size: int,
                    output_size: int,
                    lora_rank: int = 16,
                    lora_alpha: int = 1,
                    lora_dropout: float = 0.,
                    fan_in_fan_out: bool = False,
                    merge_weights: bool = True):
    if lora_rank > min(input_size, output_size):
        raise ValueError(f"LoRA rank {lora_rank} must be less or equal than {min(input_size, output_size)}")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._modules[name] = lora.Linear(input_size,
                                                output_size,
                                                r=lora_rank,
                                                lora_alpha=lora_alpha,
                                                lora_dropout=lora_dropout,
                                                fan_in_fan_out=fan_in_fan_out,
                                                merge_weights=merge_weights)
