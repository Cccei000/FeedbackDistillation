# encoding=utf-8
# python inference_llama_mp.py



"""Train"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch
import torch.distributed.run
from fengshen_inner.models.megatron import mpu
# from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM
from transformers import LlamaForCausalLM 
from transformers import  AutoTokenizer
from typing import List
import torch.nn.functional as F
import argparse
import deepspeed
import time

from chatgpt.utils import print_rank_0
from chatgpt.strategies import initialize_megatron 
from chatgpt.nn.utils import zero_pad_sequences

_MP = 4
_PP = 1
_POLICY_TOKENIZER_PATH = '/cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-c-pretrain-tokenizer/'
# _POLICY_MODEL_PATH = f'/cognitive_comp/wanghao/models/llama_sft/llama_stage_step1900_ppo_step49_MP{_MP}'
_POLICY_MODEL_PATH = '/cognitive_comp/wanghao/models/llama_sft/llama_stage_step1900_ppo_step49_hf'


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = 'left', padding_value: int = 0) -> torch.Tensor:
    assert side in ('left', 'right')
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == 'left' else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=padding_value))
    return torch.stack(padded_sequences, dim=0)

def generate(queries: List[str], tokenizer: AutoTokenizer, model: LlamaForCausalLM, apply_prefix_func=None, **generate_kwargs):
    def _apply_prefix(query):
        return f"<human>:{query.strip()}\n<bot>:"
    
    apply_prefix_func = apply_prefix_func or _apply_prefix

    def _tokenizing(queries):
        input_ids = []
        for query in queries:
            query = apply_prefix_func(query)
            input_ids.append(torch.tensor(tokenizer(query).input_ids))
        inputs = zero_pad_sequences(input_ids, side="left", padding_value=generate_kwargs["pad_token_id"])
        return inputs

    # device = next(model.parameters()).device
    # input_ids = _tokenizing(queries).to(device)
    input_ids = _tokenizing(queries).cuda()
    pad_token_id = generate_kwargs["pad_token_id"]
    # input_attention_mask = input_ids.not_equal(pad_token_id).to(dtype=torch.bool, device=device)
    input_attention_mask = input_ids.not_equal(pad_token_id).to(dtype=torch.bool).cuda()
    sequences = model.generate(
        # input_ids.to(device), attention_mask=input_attention_mask, **generate_kwargs)
        input_ids, attention_mask=input_attention_mask, **generate_kwargs)
    output = []
    for seq in sequences:
        out_text = llama_tokenizer.decode(seq.tolist()[len(input_ids[0]):], skip_special_tokens=False)
        output.append(out_text.replace('<s>','').replace('</s>',''))
    return output
    


if __name__ == "__main__":
    args = argparse.Namespace(
        policy_tokenizer_path=_POLICY_TOKENIZER_PATH,
        policy_model_path=_POLICY_MODEL_PATH,
        tensor_model_parallel_size=_MP,
        pipe_model_parallel_size=_PP,
        seed=42,
    )

    # initialize_megatron(args)

    t_start = time.time()
    # device = torch.cuda.current_device() # mpu.get_model_parallel_rank()
    llama_tokenizer = AutoTokenizer.from_pretrained(args.policy_tokenizer_path)
    model = LlamaForCausalLM.from_pretrained(
            # f"{args.policy_model_path}/part_{device}", 
            f"{args.policy_model_path}",
            device_map='auto'
            # torch_dtype=torch.bfloat16,
    ).half()
    model.eval()
    
    model = deepspeed.init_inference(model,
        replace_with_kernel_inject=True,
    )
    model = model.module

    torch.cuda.empty_cache()
    # print(f"load model from {args.policy_model_path}/part_{device}", flush=True)
    print(f"load model from {args.policy_model_path}", flush=True)

    tokenizer_vocab_size = llama_tokenizer.vocab_size
    policy_vocab_size = model.config.vocab_size
    
    if policy_vocab_size > tokenizer_vocab_size:
        bad_words_ids = [[ids] for ids in range(tokenizer_vocab_size, policy_vocab_size)]
        print_rank_0(f"BAD TOKEN IDS: {tokenizer_vocab_size}~{policy_vocab_size - 1}")
    else:
        bad_words_ids = None

    generate_kwargs = {
        "do_sample": True,
        "top_p": 1.0,   
        "top_k": 0,
        "max_length": 512,
        "repetition_penalty": 1.0,
        "temperature": 0.8,
        "bad_words_ids":bad_words_ids,
        "pad_token_id": llama_tokenizer.eos_token_id,
        "eos_token_id": llama_tokenizer.eos_token_id,
    }

    print_rank_0('START generating ....')
    print_rank_0(f"prepare model {time.time() - t_start} s elapsed, max_length {generate_kwargs['max_length']}, num_gpus {torch.cuda.device_count()}")
    queries = ['以“少爷已经十年没笑过了”开头写一个故事']*1
    ans = generate(queries=queries,
             tokenizer=llama_tokenizer,
             model=model,
             **generate_kwargs)
    print_rank_0(f'warm-up Query:{queries}\n Answer:\n{ans}')

    t_start = time.time()
    queries = ['以“少爷已经十年没笑过了”开头写一个故事，不低于3000']*1
    ans = generate(queries=queries,
             tokenizer=llama_tokenizer,
             model=model,
             **generate_kwargs)
    print_rank_0(f'Query:{queries}\n Answer:\n{ans}')
    print_rank_0(f'generated elapsed {time.time() -t_start}')

# python3 -m torch.distributed.run --nproc_per_node 2 inference_llama_mp.py
