# encoding=utf=8
import torch
import argparse
from chatgpt.utils import logging_initialize, logging_rank_0
from chatgpt.pipeline import convert_hf_to_fs_mp

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
        "--model_parallel_size", "-mp",
        type=int, default=4
    )
    parser.add_argument(
        "--is_rm", action="store_true", default=False,
        help="是否转换Reward Model"
    )
    parser.add_argument(
        "--from_hf", action="store_true", default=False,
        help="是否从Huggingface结构开始转换"
    )
    parser.add_argument(
        "--is_multi_head_rm", action="store_true", default=False,
        help="RM 是否包含多头"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None
    )
    args = parser.parse_args()
    
    logging_initialize(level="info")
    if args.is_rm:
        logging_rank_0(f"Convert Reward Model.")
    else:
        logging_rank_0(f"Convert SFT Model.")
    
    logging_rank_0(f"Is bf16: {args.dtype == 'bf16'}.")
    convert_hf_to_fs_mp(
        input_path=args.input_path,
        output_path=args.output_path,
        model_parallel_size=args.model_parallel_size,
        is_rm=args.is_rm,
        from_hf=args.from_hf,
        rm_with_multi_value_head=args.is_multi_head_rm,
        dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float16,
        model_type=args.model_type
    )
    logging_rank_0(f"Finish converting!")
