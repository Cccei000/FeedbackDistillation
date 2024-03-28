from fengshen_inner.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaModel as FengshenLlamaModel,
)
from fengshen_inner.models.megatron import mpu

from chatgpt.nn.llama.rm import (
    LlamaHFRewardModel,
    LlamaFSRewardModel,
    LlamaHFRewardModel_Mix
)

import argparse
import os
import json
import torch
from glob import glob
import copy
from tqdm import tqdm
import gc
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import PreTrainedModel,LlamaConfig,LlamaModel

from fengshen_inner.models.llama.configuration_llama import LlamaConfig as FengshenConfig
from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM as FengshenLlama
from fengshen_inner.models.megatron import mpu

__FS_FINAL_NORM_KEY__ = "model.final_layer_norm.scale"
__FS_EMBED_IN_KEY__ = "model.embed_in.word_embeddings.weight"
# __FS_EMBED_OUT_KEY__ = "embed_out.final_linear.weight"
__FS_LAYER_PREFIX__ = "model.layers"
__VALUE_HEAD_KEY__ = 'value_head.weight'
__VALUE_HEAD_BIAS_KEY__ = 'value_head.bias'


def add_args(args_parser):
    args_parser.add_argument("--pretrained_model_path", default=None, type=str)
    args_parser.add_argument("--ckpt_path", default=None, type=str)
    args_parser.add_argument("--output_path", default=None, type=str)
    args_parser.add_argument("--model_parallel_size", default=None, type=int)
    args_parser.add_argument("--is_lora", action="store_true")
    args_parser.add_argument("--multi_value_head", action="store_true")

    return args_parser

def convert_config(fs_config: FengshenConfig):
    hf_config = LlamaConfig(
        vocab_size=fs_config.vocab_size,
        hidden_size=fs_config.hidden_size,
        intermediate_size=fs_config.intermediate_size,
        num_hidden_layers=fs_config.num_hidden_layers,
        num_attention_heads=fs_config.num_attention_heads,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=fs_config.rms_norm_epsilon,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        torch_dtype=fs_config.torch_dtype,
    )
    return hf_config


def merge_data(module):
    if hasattr(module, "merge"):
        module.merge()

def convert_lora(sd,lora_rank=16):
    new_sd = {}
    for k, v in sd.items():
        if "query_key_value" not in k:
            new_sd[k]=v
        elif "base_model.weight" in k:
            lora_a = k.replace("base_model.weight","lora_A")
            lora_b = k.replace("base_model.weight","lora_B")
            v+=sd[lora_b] @ sd[lora_a] * (1.0/lora_rank)
            new_sd[k.replace("base_model.weight","weight")]=v

    return new_sd

def get_loaders(root_path, mp_size, fs_config,is_lora):
    loaders = []
    for mp in range(mp_size):
        file = os.path.join(root_path, f"mp_rank_{mp:02}_model_states.pt")
        print(f"loading {file}")
        sd = torch.load(file, map_location='cpu')
        new_sd = {}
        for k, v in sd["module"].items():
            try:
                anchor = k.index('llama')
            except:
                if 'embed_out' in k:
                    anchor = k.index('embed_out')
                else:
                    anchor = 0
            rep = k[:anchor]
            new_sd[k.replace(rep, "").replace("body", "model")] = v
        if is_lora:
            new_sd=convert_lora(new_sd)
        loaders.append(new_sd)
    return loaders

def convert(args):
    mpu.set_model_parallel_world_size(args.model_parallel_size)
    mpu.set_model_parallel_rank(0)

    fs_config = FengshenConfig.from_pretrained(args.pretrained_model_path)
    print(fs_config)
    loaded_tp_ranks = get_loaders(args.ckpt_path, args.model_parallel_size, fs_config,args.is_lora)
    print("loaded_tp_ranks 0 ",loaded_tp_ranks[0].keys())
    config = convert_config(fs_config)
    print(config)

    tokenizer = LlamaTokenizer.from_pretrained(args.pretrained_model_path)
    # num_output_shards = 1
    # num_heads_per_output_shard = config.num_attention_heads
    dims_per_head = config.hidden_size // config.num_attention_heads
    if not args.multi_value_head:
        # default setting
        hf_model = LlamaHFRewardModel(config)
    else:
        hf_model = LlamaHFRewardModel_Mix(config)

    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    dims_per_head = hidden_size // num_heads
    mp_partitions = args.model_parallel_size


    hf_model.value_head.load_state_dict(
        {
            ## if value head is nn.Linear
            # "weight": (sum([t[__VALUE_HEAD_KEY__] for t in loaded_tp_ranks])) / mp_partitions,
            # "bias": (sum([t[__VALUE_HEAD_BIAS_KEY__] for t in loaded_tp_ranks])) / mp_partitions,
            ## if value head is mpu.RowParallelLinear
            "weight": torch.cat([t[__VALUE_HEAD_KEY__] for t in loaded_tp_ranks], dim=1),
            "bias": (sum([t[__VALUE_HEAD_BIAS_KEY__] for t in loaded_tp_ranks])), # /mp_partitions
        } 
        )
    if args.multi_value_head:
        hf_model.token_value_head.load_state_dict(
            {
                ## if value head is nn.Linear
                # "weight": (sum([t[__VALUE_HEAD_KEY__] for t in loaded_tp_ranks])) / mp_partitions,
                # "bias": (sum([t[__VALUE_HEAD_BIAS_KEY__] for t in loaded_tp_ranks])) / mp_partitions,
                ## if value head is mpu.RowParallelLinear
                "weight": torch.cat([t["token_value_head.weight"] for t in loaded_tp_ranks], dim=1),
                "bias": (sum([t["token_value_head.bias"] for t in loaded_tp_ranks])), # /mp_partitions
            } 
            )        
    # EMBED_IN
    hf_model.model.embed_tokens.load_state_dict(
        {"weight": torch.cat([t[__FS_EMBED_IN_KEY__] for t in loaded_tp_ranks], dim=0)})
    # EMBED_OUT
    # hf_model.lm_head.load_state_dict(
    #     {"weight": torch.cat([t[__FS_EMBED_OUT_KEY__] for t in loaded_tp_ranks], dim=0)})
    # FINAL_LAYER_NORM
    hf_model.model.norm.load_state_dict(
        {"weight": (sum([t[__FS_FINAL_NORM_KEY__] for t in loaded_tp_ranks])) / mp_partitions})
    # layer
    for layer_i in tqdm(range(config.num_hidden_layers)):
        hf_layer = hf_model.model.layers[layer_i]
        state_dict = {}

        sharded_qkv = torch.cat(
            [t[f"{__FS_LAYER_PREFIX__}.{layer_i}.attention.query_key_value.weight"] for t in loaded_tp_ranks], dim=0)
        sharded_qkv = sharded_qkv.view(num_heads, 3, dims_per_head, hidden_size)
        q, k, v = sharded_qkv.chunk(3, dim=1)
        state_dict["self_attn.q_proj.weight"] = q.reshape(num_heads * dims_per_head, hidden_size)
        state_dict["self_attn.k_proj.weight"] = k.reshape(num_heads * dims_per_head, hidden_size)
        state_dict["self_attn.v_proj.weight"] = v.reshape(num_heads * dims_per_head, hidden_size)
        state_dict["self_attn.o_proj.weight"] = torch.cat(
            [t[f"{__FS_LAYER_PREFIX__}.{layer_i}.attention.dense.weight"] for t in loaded_tp_ranks], dim=1)
        state_dict["self_attn.rotary_emb.inv_freq"] = \
            loaded_tp_ranks[0][f"{__FS_LAYER_PREFIX__}.{layer_i}.attention.rotary_emb.inv_freq"]

        # average layernorm stats over mp ranks
        state_dict["input_layernorm.weight"] = (sum(
            [t[f"{__FS_LAYER_PREFIX__}.{layer_i}.input_layernorm.scale"] for t in loaded_tp_ranks])) / mp_partitions
        state_dict["post_attention_layernorm.weight"] = (sum(
            [t[f"{__FS_LAYER_PREFIX__}.{layer_i}.post_attention_layernorm.scale"] for t in loaded_tp_ranks])) / mp_partitions

        # mlp params
        state_dict["mlp.gate_proj.weight"] = torch.cat(
            [t[f"{__FS_LAYER_PREFIX__}.{layer_i}.mlp.w1.weight"] for t in loaded_tp_ranks], dim=0)
        state_dict["mlp.up_proj.weight"] = torch.cat(
            [t[f"{__FS_LAYER_PREFIX__}.{layer_i}.mlp.w3.weight"] for t in loaded_tp_ranks], dim=0)
        state_dict["mlp.down_proj.weight"] = torch.cat(
            [t[f"{__FS_LAYER_PREFIX__}.{layer_i}.mlp.w2.weight"] for t in loaded_tp_ranks], dim=1)

        # load state_dict into layer
        hf_layer.load_state_dict(state_dict)

    hf_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser = add_args(args_parser)
    args = args_parser.parse_args()
    print("args",args)
    convert(args=args)
    print("END")
