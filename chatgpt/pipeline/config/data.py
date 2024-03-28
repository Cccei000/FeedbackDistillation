# encoding=utf-8
from dataclasses import dataclass
from typing import Optional, Dict

from chatgpt.utils import logging_rank_0

from .base import BasicConfig


@dataclass
class DataConfig(BasicConfig):
    
    dataset_path:           Optional[str]           = None
    prefix:                 Optional[Dict[str,str]] = None
    multiturn_seperator:    Optional[str]           = "\n"
    workspace_path:         Optional[str]           = None
    
    def update(self, args: dict):
        return BasicConfig.update(self, args, "data")
    
    def check(self):
        
        check_res = True
        
        if self.dataset_path is None:
            logging_rank_0(msg=f"dataset_path is required.", level="error")
            check_res = False
        if self.workspace_path is None:
            logging_rank_0(msg=f"workspace_path is required.", level="error")
            check_res = False
        if self.prefix is not None:
            if not isinstance(self.prefix, dict):
                logging_rank_0(msg=f"Detected wrong type on 'prefix' ({type(self.prefix)}), which should be 'Dict[str,str]'", level="error")
                check_res = False
            elif self.prefix.get("model", None) is None:
                logging_rank_0(msg=f"prefix/model is required, which is the prompt of model.", level="error")
                check_res = False
                
            
        return check_res & super().check()