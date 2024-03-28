# encoding=utf-8
import argparse
from chatgpt.utils import logging_initialize, logging_rank_0
from chatgpt.pipeline import convert_fs_mp_to_hf


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", "-i",
        type=str,
    )
    parser.add_argument(
        "--output_path", "-o",
        type=str
    )
    parser.add_argument(
        "--fs_config_path", "-c",
        type=str,
        help="fengshen结构的config"
    )
    parser.add_argument(
        "--lora_rank",
        type=int, default=0,
    )
    parser.add_argument(
        "--model_parallel_size", "-mp",
        type=int, default=4
    )
    parser.add_argument(
        "--is_rm", action="store_true", default=False,
        help="是否转换Reward Model"
    )
    parser.add_argument(
        "--from_shard", action="store_true", default=False
    )
    parser.add_argument(
        "--is_multi_head_rm", action="store_true", default=False,
        help="RM 是否包含多头"
    )
    args = parser.parse_args()
    
    logging_initialize(level="info")
    if args.is_rm:
        logging_rank_0(f"Convert Reward Model.")
    else:
        logging_rank_0(f"Convert SFT Model.")
    if args.from_shard:
        logging_rank_0(f"Load From Shards of Checkpoint")
    convert_fs_mp_to_hf(
        input_path=args.input_path,
        output_path=args.output_path,
        fs_config_path=args.fs_config_path,
        model_parallel_size=args.model_parallel_size,
        is_rm=args.is_rm,
        lora_rank=args.lora_rank,
        rm_with_multi_value_head=args.is_multi_head_rm,
        from_shard=args.from_shard
    )
    logging_rank_0(f"Finish converting!")