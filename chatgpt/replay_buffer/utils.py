from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from chatgpt.nn.utils import zero_pad_sequences
from chatgpt.experience_maker import Experience,StepExperience


step_keys = ('sequences', 'action_log_probs', 'values', 'reward', 'advantages', 'attention_mask', 'action_mask', 'step_reward', 'step_value', 'step_id')
sample_keys = ('sequences', 'action_log_probs', 'values', 'reward', 'advantages', 'attention_mask', 'action_mask')
keys = ('sequences', 'action_log_probs', 'values', 'reward', 'advantages', 'attention_mask', 'action_mask', 'origin_reward', 'step_reward', 'step_value', 'step_id')


@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    values: (1)
    reward: (1) or (A)
    origin_reward: (1) or (A)
    advatanges: (1)
    attention_mask: (S)
    action_mask: (A)
    step_reward: (Step_num)
    step_value: (Step_num)
    step_id: (1)

    "A" is the number of actions.
    """
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    reward: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor] = None
    action_mask: Optional[torch.BoolTensor] = None
    step_reward: Optional[torch.Tensor] = None
    step_value: Optional[torch.Tensor] = None
    step_id: Optional[torch.Tensor] = None
    origin_reward: Optional[torch.Tensor] = None

    def to_dict(self):

        item_dict = {
            'sequences': self.sequences,
            'action_log_probs': self.action_log_probs,
            'values': self.values,
            'reward': self.reward,
            'advantages': self.advantages,
        }
        if self.attention_mask is not None:
            item_dict['attention_mask'] = self.attention_mask
        if self.action_mask is not None:
            item_dict['action_mask'] = self.action_mask
        if self.step_reward is not None:
            item_dict['step_reward'] = self.step_reward
        if self.step_value is not None:
            item_dict['step_value'] = self.step_value
        if self.step_id is not None:
            item_dict['step_id'] = self.step_id
        if self.origin_reward is not None:
            item_dict['origin_reward'] = self.origin_reward
        
        return item_dict


def split_experience_batch(experience: Experience) -> List[BufferItem]:
    batch_size = experience.sequences.size(0)
    batch_kwargs = [{} for _ in range(batch_size)]
    for key in keys:
        if not hasattr(experience, key):
            continue
        value = getattr(experience, key)
        if isinstance(value, torch.Tensor):
            vals = torch.unbind(value)
        else:
            # None
            vals = [value for _ in range(batch_size)]
        assert batch_size == len(vals), f"{batch_size}|{len(vals)}"
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v
    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


## FIXME: 不同形式的Reward在处理时或不会有问题？
def make_experience_batch(items: List[BufferItem], sequence_padding_value:int=0) -> Experience:
    kwargs = {}
    to_pad_keys = set(('sequences', 'action_log_probs', 'action_mask', 'attention_mask'))
    is_step = False
    for key in keys:
        if key == "step_reward" or key == "step_id" or key == "step_value":
            if len(items)>0 and not hasattr(items[0], key):
                is_step = True
            else:
                continue
        vals = []
        for item in items:
            if getattr(item, key) == None:
                break
            else:
                vals.append(getattr(item, key))
        if key == "sequence":
            batch_data = zero_pad_sequences(vals, "right", padding_value=sequence_padding_value)
        elif key in to_pad_keys or vals[0].ndim > 0:
            batch_data = zero_pad_sequences(vals, "right", padding_value=0)
        else:
            batch_data = torch.stack(vals, dim=0)
        kwargs[key] = batch_data

    if not is_step:
        return Experience(**kwargs)
    else:
        return StepExperience(**kwargs)

