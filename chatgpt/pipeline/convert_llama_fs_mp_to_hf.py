# encoding=utf-8
import json
import os
import time

import torch
from fengshen_inner.models.llama.configuration_llama import \
    LlamaConfig as FengshenConfig
from fengshen_inner.models.megatron import mpu
from tqdm import tqdm
from transformers import LlamaConfig
from transformers.modeling_utils import (WEIGHTS_NAME, _add_variant,
                                         shard_checkpoint)

from chatgpt.utils import logging_rank_0

__WEIGHT_MAP_FILE__ = "pytorch_model.bin.index.json"

# 区分Reward Model和Policy的前缀
RM_PREFIX = "model."
POLICY_PREFIX = "llama."


HF_FINAL_NORM_KEY = "model.norm.weight"
FS_FINAL_NORM_KEY = "final_layer_norm.scale"
HF_EMBED_IN_KEY = "model.embed_tokens.weight"
FS_EMBED_IN_KEY = "embed_in.word_embeddings.weight"


VALUE_HEAD_KEY = 'value_head.weight'
VALUE_HEAD_BIAS_KEY = 'value_head.bias'
TOKEN_VALUE_HEAD_KEY = 'token_value_head.weight'
TOKEN_VALUE_HEAD_BIAS_KEY = 'token_value_head.bias'
HF_EMBED_OUT_KEY = "lm_head.weight"
FS_EMBED_OUT_KEY = "embed_out.final_linear.weight"


FS_LAYER_PREFIX = "layers."


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
        rope_scaling=fs_config.rope_scaling,
    )
    return hf_config


def convert_lora(sd, lora_rank=16):
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

def get_loaders(root_path, mp_size, is_lora):
    loaders = []
    for mp in range(mp_size):
        file = os.path.join(root_path, f"mp_rank_{mp:02}_model_states.pt")
        logging_rank_0(f"loading {file}", "debug")
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

def get_loaders_using_from_hf_shard(root_path, mp_size, is_lora):
    loaders = []
    for mp in range(mp_size):
        path = os.path.join(root_path, f"part_{mp}")
        weight_map_file = os.path.join(path, "pytorch_model.bin.index.json")
        with open(weight_map_file, 'r') as fp:
            weight_map = json.load(fp)
        loaders_map = {}
        weight_map_with_loader = {}
        revert_weight_map = {}
        for k, v in weight_map['weight_map'].items():
            if v in revert_weight_map:
                revert_weight_map[v].append(k)
            else:
                revert_weight_map[v] = [k]
                # 打开对应的state_dict
                ld = torch.load(os.path.join(path, v), map_location='cpu')
                loaders_map[v] = ld
            weight_map_with_loader[k] = loaders_map[v]
        state_dict = {}
        for v in weight_map_with_loader.values():
            state_dict.update(v)
        if is_lora:
            state_dict = convert_lora(state_dict)
        loaders.append(state_dict)
    return loaders

def convert_fs_mp_to_hf(input_path:str,
                        output_path:str,
                        fs_config_path:str,
                        model_parallel_size:int=4,
                        is_rm:bool=False,
                        lora_rank:int=0,
                        rm_with_multi_value_head:bool=False,
                        from_shard:bool=False):
    
    logging_rank_0("Convert fs mp tp hf...")
    t1 = time.time()
    # mpu.set_model_parallel_world_size(model_parallel_size)
    # mpu.set_model_parallel_rank(0)

    fs_config = FengshenConfig.from_pretrained(fs_config_path)
    if from_shard:
        loaded_tp_ranks = get_loaders_using_from_hf_shard(input_path, model_parallel_size, lora_rank>0)
    else:
        loaded_tp_ranks = get_loaders(input_path, model_parallel_size, lora_rank>0)
    config = convert_config(fs_config)

    # num_output_shards = 1
    # num_heads_per_output_shard = config.num_attention_heads
    dims_per_head = config.hidden_size // config.num_attention_heads
    # hf_model = LlamaHFRewardModel(config)

    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    dims_per_head = hidden_size // num_heads
    mp_partitions = model_parallel_size

    concat_weight = {}
    # EMBED_IN
    concat_weight[HF_EMBED_IN_KEY] = torch.cat([t[f"{RM_PREFIX if is_rm else POLICY_PREFIX}{FS_EMBED_IN_KEY}"] for t in loaded_tp_ranks], dim=0)
    # FINAL_LAYER_NORM
    concat_weight[HF_FINAL_NORM_KEY] = (sum([t[f"{RM_PREFIX if is_rm else POLICY_PREFIX}{FS_FINAL_NORM_KEY}"] for t in loaded_tp_ranks])) / mp_partitions
    # VALUE_HEAD
    if is_rm:
        concat_weight[VALUE_HEAD_KEY] = torch.cat([t[VALUE_HEAD_KEY] for t in loaded_tp_ranks], dim=1)
        concat_weight[VALUE_HEAD_BIAS_KEY] = (sum([t[VALUE_HEAD_BIAS_KEY] for t in loaded_tp_ranks]) / mp_partitions)
        if rm_with_multi_value_head:
            concat_weight[TOKEN_VALUE_HEAD_KEY] = torch.cat([t[TOKEN_VALUE_HEAD_KEY] for t in loaded_tp_ranks], dim=1)
            concat_weight[TOKEN_VALUE_HEAD_BIAS_KEY] = (sum([t[TOKEN_VALUE_HEAD_BIAS_KEY] for t in loaded_tp_ranks]) / mp_partitions)
            
    # EMBED_OUT
    else:
        concat_weight[HF_EMBED_OUT_KEY] = torch.cat([t[FS_EMBED_OUT_KEY] for t in loaded_tp_ranks], dim=0)

    for layer_i in range(config.num_hidden_layers):
        hf_layer = f"model.layers.{layer_i}"
        fs_layer = f"{RM_PREFIX if is_rm else POLICY_PREFIX}{FS_LAYER_PREFIX}{layer_i}"

        sharded_qkv = torch.cat(
            [t[f"{fs_layer}.attention.query_key_value.weight"] for t in loaded_tp_ranks], dim=0)
        sharded_qkv = sharded_qkv.view(num_heads, 3, dims_per_head, hidden_size)
        q, k, v = sharded_qkv.chunk(3, dim=1)
        
        concat_weight[f"{hf_layer}.self_attn.q_proj.weight"] = q.reshape(num_heads * dims_per_head, hidden_size)
        concat_weight[f"{hf_layer}.self_attn.k_proj.weight"] = k.reshape(num_heads * dims_per_head, hidden_size)
        concat_weight[f"{hf_layer}.self_attn.v_proj.weight"] = v.reshape(num_heads * dims_per_head, hidden_size)

        concat_weight[f"{hf_layer}.self_attn.o_proj.weight"] = torch.cat(
            [t[f"{fs_layer}.attention.dense.weight"] for t in loaded_tp_ranks], dim=1)
        try:
            concat_weight[f"{hf_layer}.self_attn.rotary_emb.inv_freq"] = \
                loaded_tp_ranks[0][f"{fs_layer}.attention.rotary_emb.inv_freq"]
        except:
            pass
            
        # average layernorm stats over mp ranks
        concat_weight[f"{hf_layer}.input_layernorm.weight"] = (sum(
            [t[f"{fs_layer}.input_layernorm.scale"] for t in loaded_tp_ranks])) / mp_partitions
        concat_weight[f"{hf_layer}.post_attention_layernorm.weight"] = (sum(
            [t[f"{fs_layer}.post_attention_layernorm.scale"] for t in loaded_tp_ranks])) / mp_partitions

        # mlp params
        concat_weight[f"{hf_layer}.mlp.gate_proj.weight"] = torch.cat(
            [t[f"{fs_layer}.mlp.w1.weight"] for t in loaded_tp_ranks], dim=0)
        concat_weight[f"{hf_layer}.mlp.up_proj.weight"] = torch.cat(
            [t[f"{fs_layer}.mlp.w3.weight"] for t in loaded_tp_ranks], dim=0)
        concat_weight[f"{hf_layer}.mlp.down_proj.weight"] =  torch.cat(
            [t[f"{fs_layer}.mlp.w2.weight"] for t in loaded_tp_ranks], dim=1)

    weight_names = _add_variant(WEIGHTS_NAME, None)
    shards, index = shard_checkpoint(concat_weight, weights_name=weight_names)
    
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(output_path, shard_file))
    
    with open(os.path.join(output_path, __WEIGHT_MAP_FILE__), "w", encoding="utf-8") as f:
        content = json.dumps(index, indent=2, sort_keys=True) + "\n"
        f.write(content)
    config.save_pretrained(output_path)
    
    logging_rank_0(f"Use time: {round(time.time() - t1, 3)}")
    
    return
