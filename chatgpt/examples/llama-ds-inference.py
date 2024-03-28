# usage:
# deepspeed --num_gpus 8 llama-ds-inference.py
#



# This is going to improve, but at the moment, the process is a bit cumbersome - we first use
# 1. use Deepspeed-ZeRO to instantiate the model on GPUs, w/o loading the checkpoints,
# 2. free the allocated storage
# 3. start Deepspeed-Inference and only now load the checkpoint
# 4. run generate
# Done.
#


import gc
import io
import json
import math
import os
import time
import argparse
from pathlib import Path

import torch
import torch.distributed as dist

import deepspeed
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_offline_mode


# the Deepspeed team made these so it's super fast to load (~1 minute), rather than wait 10-20min loading time.
# tp_presharded_models = ["microsoft/bloom-deepspeed-inference-int8", "microsoft/bloom-deepspeed-inference-fp16"]

t_start = time.time()

# num_tokens = 100

# parser = ArgumentParser()

# parser.add_argument("--name", default='llama', type=str, help="model_name")
# parser.add_argument("--dtype", type=str, help="float16 or int8", choices=["int8", "float16"], default="float16")
# parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
# parser.add_argument("--batch_size", default=1, type=int, help="batch size")
# parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
# args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

deepspeed.init_distributed("nccl")
rank = dist.get_rank()

_POLICY_TOKENIZER_PATH = '/cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-c-pretrain-tokenizer/'
# _POLICY_MODEL_PATH = '/cognitive_comp/wanghao/models/llama_sft/llama_stage_step1900_ppo_step49_hf'
_POLICY_MODEL_PATH = f'/cognitive_comp/wanghao/models/llama_sft/llama_stage_step1900_ppo_step49_MP4'

def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)

def get_checkpoint_files(model_name_or_path):
    # extensions: .bin | .pt
    # creates a list of paths from all downloaded files in cache dir
    file_list = [str(entry) for entry in Path(model_name_or_path).rglob("*.[bp][it][n]") if entry.is_file()]
    return file_list

args = argparse.Namespace(
    policy_tokenizer_path=_POLICY_TOKENIZER_PATH,
    policy_model_path=_POLICY_MODEL_PATH,
    benchmark=True,
    batch_size=1,
)

# model_name = args.name
# infer_dtype = args.dtype

print_rank0(f"*** Loading the model {args.policy_model_path}")

tokenizer = AutoTokenizer.from_pretrained(args.policy_tokenizer_path)
config = AutoConfig.from_pretrained(args.policy_model_path)

dtype = torch.float16

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("pre-from-pretrained", force=True)

# Construct model with fake meta tensors, later will be replaced during ds-inference ckpt load
with deepspeed.OnDevice(dtype=dtype, device="meta"):
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

if args.benchmark:
    deepspeed.runtime.utils.see_memory_usage("post-from-pretrained", force=True)

model = model.eval()

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("post-init-ds-zero-init", force=True)

### Deepspeed-Inference Loading

checkpoints_json = "checkpoints.json"
def write_checkpoints_json():
    checkpoint_files = get_checkpoint_files(args.policy_model_path)
    print(checkpoint_files)
    if rank == 0:
        data = {"type": "llama", "checkpoints": checkpoint_files, "version": 1.0}
        json.dump(data, open(checkpoints_json, "w"))

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("pre-ds-inference-init", force=True)

kwargs = dict(replace_with_kernel_inject=True)

write_checkpoints_json()
dist.barrier()

model = deepspeed.init_inference(
    model,
    mp_size=world_size,
    dtype=torch.bfloat16,
    checkpoint=checkpoints_json,
    **kwargs,
)

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("post-ds-inference-init", force=True)


model = model.module

if args.benchmark:
    t_ready = time.time()


### Generate


print_rank0(f"*** Starting to generate {num_tokens} tokens with bs={args.batch_size}")

input_sentences = [
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
    "Everyone is happy and I can",
    "The new movie that got Oscar this year",
    "In the far far distance from our galaxy,",
    "Peace is the only way",
]

if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

generate_kwargs = dict(max_new_tokens=128, do_sample=True)


print_rank0(f"Generate args {generate_kwargs}")

inputs = input_sentences[: args.batch_size]


def generate():
    """returns a list of zipped inputs, outputs and number of new tokens"""

    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

    outputs = model.generate(**input_tokens, **generate_kwargs)

    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(inputs, outputs, total_new_tokens)


# warmup is a must if measuring speed as it's when all the optimizations are performed
# e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
print_rank0("*** Running generate warmup")
_ = generate()

print_rank0("*** Running generate")
t_generate_start = time.time()
generated = generate()
t_generate_span = time.time() - t_generate_start
for i, o, _ in generated:
    print_rank0(f"{'-'*60}\nin={i}\nout={o}\n")

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("end-of-run", force=True)

### Benchmark

# benchmark it!
if args.benchmark:
    print_rank0("*** Running benchmark")

    # warm up
    for i in range(1):
        _ = generate()
    torch.cuda.synchronize()

    # benchmark
    t0 = time.time()
    cycles = 5
    total_new_tokens_generated = 0
    for i in range(cycles):
        generated = generate()
        total_new_tokens_generated += sum(new_tokens for _, _, new_tokens in generated)
    torch.cuda.synchronize()
    throughput = (time.time() - t0) / (total_new_tokens_generated)
    print_rank0(
        f"""
*** Performance stats:
Throughput per token including tokenize: {throughput*1000:.2f} msecs
Start to ready to generate: {t_ready - t_start:.3f} secs
Tokenize and generate {total_new_tokens_generated} (bs={args.batch_size}) tokens: {t_generate_span:.3f} secs
Start to finish: {t_ready - t_start + t_generate_span:.3f} secs
"""
    )
