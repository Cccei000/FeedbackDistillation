from typing import Optional, Tuple, Union, Callable, Dict
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class Reflector(nn.Module):
    """
    Base class for reflection model (also named feedback model, for generating feedback).

    Args:
        model (nn.Module): body model for the reflector.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @abstractmethod
    def reflect(self, inputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs: dict containing fields 'sequences', 'attention_mask', 'action_mask'

        Returns: 
            dict containing fields 'sequences', 'query_mask', 'response_mask', 'feeedback_mask'
    
        """
        pass
        
    @abstractmethod
    def generate(self, input_ids, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    @property
    def device(self):
        return self.model.device
