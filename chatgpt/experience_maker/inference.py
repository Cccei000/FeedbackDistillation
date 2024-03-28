import torch
import gc
from typing import List, Dict, Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from chatgpt.nn import Actor
from tqdm import tqdm
from collections import defaultdict
from chatgpt.nn.utils import compute_reward, compute_approx_kl
from chatgpt.utils import is_rank_0
# from chatgpt.trainer import get_rank

from chatgpt.experience_maker import ExperienceMaker


class InferenceExperienceMaker(ExperienceMaker):
    """
    Naive experience maker.
    """

    def __init__(self,
        actor: Actor,
        reward_model: nn.Module,
        seed: int = 1234,
        pad_token_id: int = 0,
        rm_minibatch_size: int = 1,
        gen_minibatch_size: Optional[int] = None,
        enable_gae:bool=False) -> None:
        super().__init__(actor, None, reward_model, None, 0.0)
        self.gen_minibatch_size = gen_minibatch_size
        self.rm_minibatch_size = rm_minibatch_size
        self.pad_token_id = pad_token_id
        self.seed = seed
        self.enable_gae = enable_gae
        print(f"Rank-{dist.get_rank()}: {seed}")
        torch.manual_seed(self.seed)
        return

    def make_experience_with_actor(self, input_ids: torch.Tensor, **generate_kwargs) -> Dict[str, torch.Tensor]:
        
        mini_batch_dict = defaultdict(list)
        mini_batch_response = {}
        
        with torch.no_grad():
            # 生成文本
            gen_batch_size = self.gen_minibatch_size if self.gen_minibatch_size else self.actor_minibatch_size
            for i in tqdm(range(0, len(input_ids), gen_batch_size),
                          desc=f"Generating",
                          disable=not is_rank_0()):
                mini_batch_input = input_ids[i: i + gen_batch_size]
                sequences, attention_mask, action_mask, _ = self.actor.module.generate(mini_batch_input,
                                                                                    return_action_mask=True,
                                                                                    **generate_kwargs)
                gc.collect()
                
                mini_batch_dict["sequence"].append(sequences)               # (bs, seq_len)
                mini_batch_dict["attention_mask"].append(attention_mask)    # (bs, seq_len)
                mini_batch_dict["action_mask"].append(action_mask)          # (bs, seq_len)
            
            # 对齐生成的文本（统一右填充）
            for key, value in mini_batch_dict.items():
                max_len = max(item.shape[1] for item in value)
                if key == "sequence":
                    padded_value = [
                        F.pad(item, (0, max_len - item.shape[1]), value=self.pad_token_id) # 使用传入的pad_token_id填充
                        for item in value
                    ]
                else:
                    padded_value = [
                        F.pad(item, (0, max_len - item.shape[1]), value=0)
                        for item in value
                    ]
                mini_batch_response[key] = torch.cat(padded_value, dim=0) # (num, seq_len)

        return mini_batch_response

    def make_experience_with_reward_model(self, mini_batch_response:Dict[str,torch.Tensor]) -> torch.Tensor:
        
        self.reward_model.cuda()
        total_num = mini_batch_response["sequence"].shape[0]
        rewards = []

        with torch.no_grad():
            for i in tqdm(range(0, total_num, self.rm_minibatch_size), 
                          desc=f"Scoring",
                          disable=not is_rank_0()):

                start, end = i, i + self.rm_minibatch_size
                sequences = mini_batch_response["sequence"][start : end]                            # (bs, seq_len)
                attention_mask = mini_batch_response["attention_mask"][start : end]                 # (bs, seq_len)
                action_mask = mini_batch_response["action_mask"][start : end]                       # (bs, seq_len)

                r = self.reward_model(sequences, action_mask, attention_mask) # (bs,)
                reward = compute_reward(r, 0.0, None, None, action_mask=action_mask)
                rewards.append(reward)
                
        rewards = torch.cat(rewards)
        self.reward_model.cpu()

        return rewards

    @torch.no_grad()
    def make_experience(self, input_ids: torch.Tensor, **generate_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:

        self.actor.eval()
        self.reward_model.eval().cpu()

        # 先做actor critic forward
        mini_batch_response = self.make_experience_with_actor(input_ids=input_ids, **generate_kwargs)
        gc.collect()
        rewards = self.make_experience_with_reward_model(mini_batch_response)
        gc.collect()
        
            
        return mini_batch_response["sequence"], rewards
    
    def logging(self):
        return
