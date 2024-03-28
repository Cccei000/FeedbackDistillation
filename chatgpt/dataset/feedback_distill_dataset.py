# encoding-utf-8
import torch
from torch.utils.data import Dataset

from chatgpt.replay_buffer import FDReplayBuffer
from chatgpt.utils import LoggingLevel, logging_rank_0


class FeedbackDistillDataset(Dataset):

    def __init__(self, replay_buffer:FDReplayBuffer) -> None:
        super().__init__()
        self.replay_buffer = replay_buffer
        self.collate_fn = self.replay_buffer.collate_fn

    def __len__(self):
        return len(self.replay_buffer)

    def __getitem__(self, index: int):
        return self.replay_buffer[index]