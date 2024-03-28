# encoding=utf-8
from chatgpt.utils import logging_initialize, logging_rank_0, is_rank_0



def test():
    logging_initialize(level="debug")
    logging_rank_0(f"Rank0:{is_rank_0()}.", level="debug")
    logging_rank_0(f"Rank0:{is_rank_0()}.", level="info")
    logging_rank_0(f"Rank0:{is_rank_0()}.", level="warning")
    logging_rank_0(f"Rank0:{is_rank_0()}.", level="error")
    logging_rank_0(f"Rank0:{is_rank_0()}.", level="critical")
    
    return

if __name__ == "__main__":
    test()