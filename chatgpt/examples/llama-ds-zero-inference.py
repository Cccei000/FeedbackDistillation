# usage:
# deepspeed --num_gpus 8 bloom-ds-zero-inference.py --name bigscience/bloom
#
# to run benchmarks:
# deepspeed --num_gpus 8 bloom-ds-zero-inference.py --name bigscience/bloom --benchmark
#


# This is going to improve, but at the moment, the process is a bit cumbersome - we first use
# 1. use Deepspeed-ZeRO to instantiate the model on GPUs, w/o loading the checkpoints,
# 2. free the allocated storage
# 3. start Deepspeed-Inference and only now load the checkpoint
# 4. run generate
# Done.
#


import gc
import math
import os
import time
import argparse
import torch
import torch.distributed as dist

from typing import List
import torch.nn.functional as F
import deepspeed
from transformers import AutoConfig, LlamaForCausalLM, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock

_POLICY_TOKENIZER_PATH = '/cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-c-pretrain-tokenizer/'
_POLICY_MODEL_PATH = '/cognitive_comp/wanghao/models/llama_sft/llama_stage_step1900_ppo_step49_hf'

t_start = time.time()


local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

deepspeed.init_distributed("nccl")
rank = dist.get_rank()


def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)

args = argparse.Namespace(
    policy_tokenizer_path=_POLICY_TOKENIZER_PATH,
    policy_model_path=_POLICY_MODEL_PATH,
    benchmark=True,
    batch_size=1,
    cpu_offload=False,
    num_tokens=1024
)

### Model loading and instantiating on GPU (via ZeRO)

print_rank0(f"*** Loading the model {args.policy_model_path}")

tokenizer = AutoTokenizer.from_pretrained(args.policy_tokenizer_path)
config = AutoConfig.from_pretrained(args.policy_model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id

dtype = torch.bfloat16 
model_hidden_size = config.hidden_size
train_batch_size = 1 * world_size

ds_config = {
    "fp16": {
        "enabled": dtype == torch.float16,
    },
    "bf16": {
        "enabled": dtype == torch.bfloat16,
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 0,
    },
    'train_batch_size': train_batch_size
}

if args.cpu_offload:
    ds_config["zero_optimization"]["offload_param"] = dict(device="cpu", pin_memory=True)


dschf = HfDeepSpeedConfig(ds_config)  # this tells from_pretrained to instantiate directly on gpus

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("pre-from-pretrained", force=True)

model = LlamaForCausalLM.from_pretrained(args.policy_model_path, torch_dtype=torch.bfloat16)

if args.benchmark:
    deepspeed.runtime.utils.see_memory_usage("post-from-pretrained", force=True)

model = model.eval()

print_rank0(ds_config)

ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()
model = ds_engine.module

if args.benchmark:
    t_ready = time.time()
    deepspeed.runtime.utils.see_memory_usage("start-of-generate", force=True)


### Generate

print_rank0(f"*** Starting to generate {args.num_tokens} tokens with bs={args.batch_size}")

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

generate_kwargs = dict(max_new_tokens=args.num_tokens, do_sample=True, pad_token_id=tokenizer.eos_token_id)
# Important: if using multiple unique streams to avoid hanging if one generation finished early - one must also add:
# generate_kwargs.update(synced_gpus=True)

print_rank0(f"Generate args {generate_kwargs}")
inputs = input_sentences[: args.batch_size]

def zero_pad_sequences(sequences: List[torch.Tensor], side: str = 'left', padding_value: int = 0) -> torch.Tensor:
    assert side in ('left', 'right')
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == 'left' else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=padding_value))
    return torch.stack(padded_sequences, dim=0)

def generate():
    """returns a list of zipped inputs, outputs and number of new tokens"""
    
    device=torch.cuda.current_device()

    def _apply_prefix(query):
        return f"<human>:{query.strip()}\n<bot>:"
    
    apply_prefix_func =  _apply_prefix

    def _tokenizing(queries):
        input_ids = []
        for query in queries:
            query = apply_prefix_func(query)
            input_ids.append(torch.tensor(tokenizer(query).input_ids))
        inputs = zero_pad_sequences(input_ids, side="left", padding_value=generate_kwargs["pad_token_id"])
        return inputs

    device = next(model.parameters()).device
    input_ids = _tokenizing(inputs).to(device)
    pad_token_id = generate_kwargs["pad_token_id"]
    input_attention_mask = input_ids.not_equal(pad_token_id).to(dtype=torch.bool, device=device)
    sequences = model.generate(
        input_ids.to(device), attention_mask=input_attention_mask, **generate_kwargs)
    output = []
    for seq in sequences:
        out_text = llama_tokenizer.decode(seq.tolist()[len(input_ids[0]):], skip_special_tokens=False)
        output.append(out_text.replace('<s>','').replace('</s>',''))
    return (inputs, output, [0]*len(output))
    # input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True).input_ids
    # input_tokens = input_tokens.to(device)
    # print(input_tokens)

    # pad_token_id = generate_kwargs["pad_token_id"]
    # input_attention_mask = input_tokens.not_equal(pad_token_id).to(dtype=torch.bool, device=device)

    # outputs = model.generate(input_tokens, attention_mask=input_attention_mask, **generate_kwargs)

    # input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    # output_tokens_lengths = [x.shape[0] for x in outputs]

    # total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
    # outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # return zip(inputs, outputs, total_new_tokens)


# XXX: this is currently doing world_size streams on world_size gpus, so we can feed it different inputs on each! and hence the time can be divided by world_size

print_rank0("*** Running generate")
t_generate_start = time.time()
pairs = generate()
t_generate_span = time.time() - t_generate_start
for i, o, _ in pairs:
    print_rank0(f"{'-'*60}\nin={i}\nout={o}\n")


### Benchmark

if args.benchmark:
    # clear cache / free memory
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("end-of-generate", force=True)

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
    # note that we actually generate world_size unique streams (though the benchmark feeds the same inputs)
    total_new_tokens_generated *= world_size
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
