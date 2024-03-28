from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from chatgpt.experience_maker import Experience


class ReplayBuffer(ABC):
    """Replay buffer base class. It stores experience.

     Args:
         sample_batch_size (int): Batch size when sampling.
         limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
    """

    def __init__(self, sample_batch_size: int, limit: int = 0) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit

    @abstractmethod
    def append(self, experiences: List[Experience]) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def sample(self) -> Experience:
        pass

    @abstractmethod
    def get_advantage_statistics(self,  is_distributed:bool=True, eps:float=1e-8) -> Tuple[float, float]:
        pass
    
    @abstractmethod
    def update_with_gae(self, gamma:float=1.0, lam:float=0.95, is_distributed:bool=True, eps:float=1e-8) -> Tuple[float, float]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        pass

    @abstractmethod
    def collate_fn(self, batch: Any) -> Experience:
        pass
