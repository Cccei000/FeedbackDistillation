# encoding=utf-8
import torch.distributed as dist
from typing import Dict, List, Optional, Union

from chatgpt.utils import logging_rank_0
from deepspeed.profiling.flops_profiler import FlopsProfiler


GENERATE = "infer_generate"
ACTOR_INFER = "infer_actor"
CRITIC_INFER = "infer_critic"
REF = "infer_initial"
RM = "infer_rm"
ACTOR_TRAIN = "train_actor"
CRITIC_TRAIN = "train_critic"

INFER_MODELS = [GENERATE, ACTOR_INFER, CRITIC_INFER, REF, RM]
TRAIN_MODELS = [ACTOR_TRAIN, CRITIC_TRAIN]
ALL_MODELS = INFER_MODELS + TRAIN_MODELS


class FlopsTimer:
    
    def __init__(self,
                 hidden_size:int,
                 num_layers:int,
                 total_params:int,
                 world_size:int,
                 vocab_size:int,
                 is_train:bool=False,):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.total_params = total_params
        self.vocab_size = vocab_size
        self.counter = 0
        self.timer = 0
        self.factor = 3 if is_train else 1
        self._world_size = world_size
        logging_rank_0(f"world size: {self.world_size}", "debug")
        return
    
    def calculate(self, seq_length:int, batch_size:int, is_cached_generate:bool=False) -> int:
        
        total_params = self.total_params
        hidden_size = self.hidden_size
        num_layers = self.num_layers
        vocab_size = self.vocab_size
        
        # 区别计算use cache生成和其他的计算量
        if is_cached_generate:
            ff = batch_size * total_params * 2                              # 2 * b * param
            attn = 4 * batch_size * seq_length * hidden_size * num_layers   # 4 * b * s * h * l
            vocab = 2 * batch_size * hidden_size * vocab_size               # 2 * b * h * V
        else:
            ff = batch_size * seq_length * total_params * 2                             # 2 * b * s * param
            attn = 4 * batch_size * seq_length * seq_length * hidden_size * num_layers  # 4 * b * s^2 * h * l
            vocab = 2 * batch_size * hidden_size * seq_length * vocab_size              # 2 * b * s * h * V
            
        return (ff + attn + vocab) * self.factor
    
    def update_calculation(self, seq_length:Optional[int]=None, batch_size:Optional[int]=None, batch_info:Optional[List[List[int]]]=None):
        """增加计算总量

        Args:
            seq_length (Optional[int], optional): _description_. Defaults to None.
            batch_size (Optional[int], optional): _description_. Defaults to None.
            batch_info (Optional[List[List[int]]], optional): _description_. Defaults to None.
        """
        
        if seq_length is not None and batch_size is not None:
            self.counter += self.calculate(seq_length=seq_length, batch_size=batch_size)
        elif batch_info is not None:
            for info in batch_info:
                batch_size, seq_length, is_cached_generate = info[0], info[1], info[2]
                self.counter += self.calculate(seq_length=seq_length, batch_size=batch_size, is_cached_generate=is_cached_generate)
        else:
            logging_rank_0(f"Fail to update calculation!", "debug")
        
        return

    def update_timer(self, iter_time_s):
        """增加计算耗时

        Args:
            iter_time_s (_type_): _description_
        """
        self.timer += iter_time_s
        return 
    
    def reset(self):
        """重置
        """
        self.timer = 0
        self.counter = 0 
    
    def get_flops(self):
        """获取计算效率

        Returns:
            _type_: _description_
        """
        return self.counter / (self.timer * self._world_size) if self.timer > 0 else 0.0
    
    @property
    def total_calculation(self):
        return self.counter
    
    @property
    def total_time(self):
        return self.timer
    
    @property
    def world_size(self):
        return self._world_size
    
    @world_size.setter
    def world_size(self, n):
        self._world_size = n
    
    
class FlopsTimerGroup:
    
    def __init__(self, timers:Dict[str, FlopsTimer]) -> None:
        self.timers:Dict[str, FlopsTimer] = timers
        self.models = list(timers.keys())
        return
    
    def update_calculation(self, model_name:str, seq_length:Optional[int]=None, batch_size:Optional[int]=None, batch_info:Optional[List[List[int]]]=None) -> None:
        
        if model_name in self.models:
            prev = self.timers[model_name].counter
            self.timers[model_name].update_calculation(batch_size=batch_size, seq_length=seq_length, batch_info=batch_info)
            # logging_rank_0(f"{model_name}: {seq_length} | {batch_size} add cul {self.timers[model_name].counter - prev}", "debug")
        else:
            logging_rank_0(f"model_name ({model_name}) isn't included in {self.models}", "debug")
            
        return
    
    def update_timer(self, model_name:str, iter_time_s) -> None:
        
        if model_name in self.models:
            self.timers[model_name].update_timer(iter_time_s)
        else:
            logging_rank_0(f"model_name ({model_name}) isn't included in {self.models}", "debug")
        return
    
    def reset(self, model_name:str) -> None:
        if model_name in self.models:
            self.timers[model_name].reset()
        else:
            logging_rank_0(f"model_name ({model_name}) isn't included in {self.models}", "debug")
        return
        
    def reset_all(self) -> None:
        for timer in self.timers.values():
            timer.reset()
        
        return
    
    def get_flops(self, model_name:Optional[Union[str, List[str]]]=None) -> Dict[str, float]:
        
        if model_name is None:
            return {
                name: timer.get_flops()
                for name, timer in self.timers.items()
            }
            
        if isinstance(model_name, str):
            model_name = [model_name]
            
        if isinstance(model_name, list):
            output = {}
            for name in model_name:
                if isinstance(name, str) and name in self.models:
                    # logging_rank_0(f"{name}: flop-{self.timers[name].counter} | time-{self.timers[name].timer}", "debug")
                    output[name] = self.timers[name].get_flops()
                elif isinstance(name, str) and name not in self.models:
                    logging_rank_0(f"model_name ({name}) isn't included in {self.models}", "debug")
            
            return output
        
        return {}
    
    def get_avg_flops(self, model_name:List[str]) -> float:
        total_time = 0
        total_tokens = 0
        world_size = 1
        for name in model_name:
            if isinstance(name, str) and name in self.models:
                total_time += self.timers[name].total_time
                total_tokens += self.timers[name].total_calculation
                world_size = self.timers[name].world_size
            elif isinstance(name, str) and name not in self.models:
                logging_rank_0(f"model_name ({name}) isn't included in {self.models}", "debug")
            
        return total_tokens / (total_time * world_size) if total_time != 0 else 0.0 
    
    
    
class DeepspeedFlopsTimerGroup:
    
    def __init__(self, timers:Dict[str, FlopsProfiler]) -> None:
        self.timers:Dict[str, FlopsProfiler] = timers
        self.models = list(timers.keys())
        self.record_flops:Dict[str, float] = dict(zip(self.models, [0.0]*len(self.models)))
        return
    
    def start_profile(self, model_name:str) -> None:
        if model_name in self.models:
            self.timers[model_name].start_profile()
        # else:
        #     logging_rank_0(f"model_name ({model_name}) isn't included in {self.models}", "debug")
        return
    
    def stop_profile(self, model_name:str) -> None:
        if model_name in self.models:
            self.timers[model_name].stop_profile()
        # else:
        #     logging_rank_0(f"model_name ({model_name}) isn't included in {self.models}", "debug")
        return

    def end_profile(self, model_name:str) -> None:
        if model_name in self.models:
            self.timers[model_name].end_profile()
        # else:
        #     logging_rank_0(f"model_name ({model_name}) isn't included in {self.models}", "debug")
        return
    
    def get_flops(self, model_name:Optional[Union[str, List[str]]]=None) -> Dict[str, float]:
        
        if model_name is None:
            return {
                name: timer.get_total_flops(as_string=False)
                for name, timer in self.timers.items()
            }
            
        if isinstance(model_name, str):
            model_name = [model_name]
            
        if isinstance(model_name, list):
            output = {}
            for name in model_name:
                if isinstance(name, str) and name in self.models:
                    output[name] = self.timers[name].get_total_flops(as_string=False)
                # elif isinstance(name, str) and name not in self.models:
                #     logging_rank_0(f"model_name ({name}) isn't included in {self.models}", "debug")
            
            return output
        
        return {}
    
    def get_avg_flops(self, model_name:List[str]) -> float:
        all_flops = []
        for name in model_name:
            if isinstance(name, str) and name in self.models:
                flops = self.timers[name].get_total_flops()
                all_flops.append(flops)
            # elif isinstance(name, str) and name not in self.models:
            #     logging_rank_0(f"model_name ({name}) isn't included in {self.models}", "debug")
            
        return sum(all_flops) / len(all_flops) if len(all_flops) != 0 else 0.0 