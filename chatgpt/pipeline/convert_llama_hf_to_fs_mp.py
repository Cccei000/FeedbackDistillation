# encoding=utf-8
import json
import os
import time
from typing import Optional

import torch
from fengshen_inner.models.llama.configuration_llama import \
    LlamaConfig as FengshenConfig
# from fengshen_inner.models.megatron.layers.word_embeddings import Embedding
from torch.nn import Embedding
from transformers.modeling_utils import (WEIGHTS_NAME, _add_variant,
                                         shard_checkpoint)
from transformers.models.llama import LlamaConfig

from chatgpt.models.baichuan import BaichuanConfig
from chatgpt.pipeline.config import ModelConvertPipelineConfig
from chatgpt.utils import logging_rank_0

### Policy
__HF_NORM_PREFIX__ = "llama.final_layer_norm"
__HF_EMBED_IN_KEY__ = "llama.embed_in.word_embeddings.weight"
__HF_EMBED_OUT_KEY__ = "embed_out.final_linear.weight"
__HF_LAYER_PREFIX__ = "llama.layers"
__WEIGHT_MAP_FILE__ = "pytorch_model.bin.index.json"


### RM
__HF_VALUE_HEAD_KEY__ = "value_head.weight"
__HF_VALUE_HEAD_BIAS_KEY__ = "value_head.bias"
__HF_TOKEN_VALUE_HEAD_KEY__ = "token_value_head.weight"
__HF_TOKEN_VALUE_HEAD_BIAS_KEY__ = "token_value_head.bias"


LLAMA_HF_TO_FS = {
    # embed_in
    "model.embed_tokens.weight": "llama.embed_in.word_embeddings.weight",
    # final_norm
    "model.norm.weight": "llama.final_layer_norm.scale",
}


LLAMA_POLICY_LM_HEAD_TO_FS = {
    # lm_head
    "lm_head.weight": "embed_out.final_linear.weight",
}


LLAMA_RM_VALUE_HEAD_TO_FS = {
    # rm value_head
    "value_head.weight": "value_head.weight",
    "value_head.bias": "value_head.bias",
}


LLAMA_RM_MULTI_VALUE_HEAD_TO_FS = {
    # rm value_head
    "value_head.weight": "value_head.weight",
    "value_head.bias": "value_head.bias",
    # rm token_value_head
    "token_value_head.weight": "token_value_head.weight",
    "token_value_head.bias": "token_value_head.bias",
}


LLAMA_HF_LAYER_TO_FS = {
    ".self_attn.o_proj.weight": ".attention.dense.weight",
    ".mlp.gate_proj.weight": ".mlp.w1.weight",
    ".mlp.down_proj.weight": ".mlp.w2.weight",
    ".mlp.up_proj.weight": ".mlp.w3.weight",
    ".input_layernorm.weight": ".input_layernorm.scale",
    ".post_attention_layernorm.weight": ".post_attention_layernorm.scale",
    ".self_attn.rotary_emb.inv_freq": ".attention.rotary_emb.inv_freq"
}


def convert_llama_config(hf_config: LlamaConfig, dtype):
    try:
        rope_scaling = hf_config.rope_scaling
    except AttributeError:
        rope_scaling = None
    
    fs_config = FengshenConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        intermediate_size=hf_config.intermediate_size,
        hidden_act=hf_config.hidden_act,
        rotary_pct=1,
        rotary_emb_base=10000,
        max_position_embeddings=hf_config.max_position_embeddings,
        initializer_range=hf_config.initializer_range,
        rms_norm_epsilon=hf_config.rms_norm_eps,
        torch_dtype=dtype,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_parallel_residual=False,
        rope_scaling=rope_scaling,
    )
    fs_config.llama_mlp_multiple_of = 256
    assert fs_config.intermediate_size % fs_config.llama_mlp_multiple_of == 0, \
        f"{fs_config.intermediate_size} % {fs_config.llama_mlp_multiple_of}"
    fs_config.init_method = "small_init"
    fs_config.hidden_dropout = 0
    fs_config.output_layer_init_method = "wang_init"
    fs_config.pos_emb = "rotary"
    fs_config.norm = "rmsnorm"
    fs_config.gpt_j_residual = False
    fs_config.gpt_j_tied = False
    fs_config.apply_query_key_layer_scaling = False
    fs_config.attention_softmax_in_fp32 = False
    fs_config.scaled_masked_softmax_fusion = True
    fs_config.scaled_upper_triang_masked_softmax_fusion = False
    fs_config.bias_gelu_fusion = False
    fs_config.attention_dropout = 0
    fs_config.output_layer_parallelism = "column"
    fs_config.eod_mask_loss = False
    fs_config.bias_dropout_fusion = False
    fs_config.attention_config = [[["flash"], "all"]]
    fs_config.mlp_type = "llama"
    fs_config.use_bias_in_attn_linear = False
    fs_config.lora = False
    return fs_config


def convert_baichuan_config(hf_config: BaichuanConfig, dtype):
    
    fs_config = FengshenConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        intermediate_size=hf_config.intermediate_size,
        hidden_act=hf_config.hidden_act,
        rotary_pct=None,
        rotary_emb_base=None,
        max_position_embeddings=hf_config.model_max_length,
        initializer_range=hf_config.initializer_range,
        rms_norm_epsilon=hf_config.rms_norm_eps,
        torch_dtype=dtype,
        use_cache=True,
        pad_token_id=hf_config.pad_token_id,
        bos_token_id=hf_config.bos_token_id,
        eos_token_id=hf_config.eos_token_id,
        tie_word_embeddings=hf_config.tie_word_embeddings,
        use_parallel_residual=False,
        rope_scaling=None,
    )
    fs_config.llama_mlp_multiple_of = 128
    assert fs_config.intermediate_size % fs_config.llama_mlp_multiple_of == 0, \
        f"{fs_config.intermediate_size} % {fs_config.llama_mlp_multiple_of}"
    fs_config.init_method = "small_init"
    fs_config.hidden_dropout = 0
    fs_config.output_layer_init_method = "wang_init"
    fs_config.pos_emb = "alibi"
    fs_config.norm = "rmsnorm"
    fs_config.gpt_j_residual = False
    fs_config.gpt_j_tied = False
    fs_config.apply_query_key_layer_scaling = False
    fs_config.attention_softmax_in_fp32 = False
    fs_config.scaled_masked_softmax_fusion = True
    fs_config.scaled_upper_triang_masked_softmax_fusion = False
    fs_config.bias_gelu_fusion = False
    fs_config.attention_dropout = 0
    fs_config.output_layer_parallelism = "column"
    fs_config.eod_mask_loss = False
    fs_config.bias_dropout_fusion = False
    fs_config.attention_config = [[["global"], "all"]]
    fs_config.mlp_type = "llama"
    fs_config.use_bias_in_attn_linear = False
    fs_config.lora = False
    return fs_config


def make_output_dir(path, parallel_size):
    """
    root_dir
    |--- part_0
    |___ part_1
    """
    try:
        os.mkdir(path)
    except:
        pass

    for i in range(parallel_size):
        try:
            os.mkdir(os.path.join(path, f"part_{i}"))
        except:
            pass


def find_closest_multiple(current_num, n):
    if current_num % n == 0:
        return current_num
    closest_multiple = ((current_num // n) + 1) * n
    return closest_multiple


def resize_token_embeddings(prev_embed:Embedding, new_num_tokens: int, config:FengshenConfig):
    old_vocab_size = config.vocab_size
    config.vocab_size = new_num_tokens
    
    # new_embed_in = Embedding(config,
    #                             config.hidden_size,
    #                             config.vocab_size,
    #                             config.max_position_embeddings,
    #                             config.hidden_dropout,
    #                             init_method=lambda x: torch.nn.init.xavier_normal_(x),
    #                             num_tokentypes=0)
    new_embed_in = Embedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
    )
    new_embed_in.weight.data[:old_vocab_size, :] = prev_embed.weight.data[:old_vocab_size, :]
    return new_embed_in.weight.data, config


def get_loaders(root_dir, weight_map):
    loaders_map = {}
    weight_map_with_loader = {}
    revert_weight_map = {}
    for k, v in weight_map['weight_map'].items():
        if v in revert_weight_map:
            revert_weight_map[v].append(k)
        else:
            revert_weight_map[v] = [k]
            # 打开对应的state_dict
            ld = torch.load(os.path.join(root_dir, v), map_location='cpu')
            loaders_map[v] = ld
        weight_map_with_loader[k] = loaders_map[v]
    return weight_map_with_loader, revert_weight_map, loaders_map.values()


def save_splits(weight_map, output_dir, helper, config):
    for rank, sd in enumerate(helper.sequential_cache):
        output_part_dir = os.path.join(output_dir, f"part_{rank}")
        with open(os.path.join(output_part_dir, __WEIGHT_MAP_FILE__), 'w') as f:
            json.dump(weight_map, f)
        config.save_pretrained(output_part_dir)
        for file_name, keys in helper.revert_weight_map.items():
            output_sd = {}
            for k in keys:
                if k in sd:
                    output_sd[k] = sd[k]
            torch.save(output_sd, os.path.join(output_part_dir, file_name))


class Helper:
    def __init__(self, model_parallel_size:int, input_path:str):
        self.num_output_shards = model_parallel_size
        self.sequential_cache = [{} for _ in range(model_parallel_size)]
        
        weight_map_file = os.path.join(input_path, __WEIGHT_MAP_FILE__)
        with open(weight_map_file, 'r') as fp:
            weight_map = json.load(fp)
        self.weight_map, self.revert_weight_map, self.loaders = get_loaders(
            input_path, weight_map)
        
    
    def convert_hf_to_fs(self, hf_config, fs_config, multiplier:int=1, is_rm:bool=False, rm_with_multi_value_head:bool=False, is_baichuan:bool=False):
        
        # mpu.set_model_parallel_world_size(1)
        # mpu.set_model_parallel_rank(0)
        # mpu.set_init_params_in_cuda(False)
        
        num_heads = hf_config.num_attention_heads
        hidden_size = hf_config.hidden_size
        dims_per_head = hidden_size // num_heads
        
        concat_weight = {}
        for value in self.loaders:
            concat_weight.update(value)
        for prev, curr in LLAMA_HF_TO_FS.items():
            concat_weight[curr] = concat_weight[prev]
            if curr != prev:
                del concat_weight[prev]

        # 根据不同的模型类型，转换对应的输出层
        if is_rm and rm_with_multi_value_head:
            embed_out_map = LLAMA_RM_MULTI_VALUE_HEAD_TO_FS
        elif is_rm:
            embed_out_map = LLAMA_RM_VALUE_HEAD_TO_FS
        else:
            embed_out_map = LLAMA_POLICY_LM_HEAD_TO_FS
            
        for prev, curr in embed_out_map.items():
            concat_weight[curr] = concat_weight[prev]
            if curr != prev:
                del concat_weight[prev]
            

        # prev_embed = Embedding(fs_config,
        #                     fs_config.hidden_size,
        #                     fs_config.vocab_size,
        #                     fs_config.max_position_embeddings,
        #                     fs_config.hidden_dropout,
        #                     init_method=lambda x: torch.nn.init.xavier_normal_(x),
        #                     num_tokentypes=0)
        prev_embed = Embedding(
            num_embeddings=fs_config.vocab_size,
            embedding_dim=fs_config.hidden_size,
        )
        embed_state_dict = {
            __HF_EMBED_IN_KEY__[len("llama.embed_in.word_embeddings."):]: concat_weight[__HF_EMBED_IN_KEY__]
        }
        prev_embed.load_state_dict(embed_state_dict)
        concat_weight[__HF_EMBED_IN_KEY__], fs_config = resize_token_embeddings(prev_embed, find_closest_multiple(fs_config.vocab_size, int(multiplier)), fs_config)
        
        for layer_i in range(fs_config.num_hidden_layers):
            fs_layer = f"llama.layers.{layer_i}"
            hf_layer = f"model.layers.{layer_i}"
            
            for prev, curr in LLAMA_HF_LAYER_TO_FS.items():
                try:
                    fs_key = fs_layer + curr
                    hf_key = hf_layer + prev
                    concat_weight[fs_key] = concat_weight[hf_key]
                    if fs_key != hf_key:
                        del concat_weight[hf_key]
                except KeyError:
                    logging_rank_0(f"miss key {hf_key}", "debug")
            
            if is_baichuan:
                baichuan_qkv = concat_weight[hf_layer + ".self_attn.W_pack.weight"] # (3*h, h)
                baichuan_qkv = baichuan_qkv.unflatten(0, (3, hidden_size))
                w_q = baichuan_qkv[0].view(num_heads, dims_per_head, hidden_size)
                w_k = baichuan_qkv[1].view(num_heads, dims_per_head, hidden_size)
                w_v = baichuan_qkv[2].view(num_heads, dims_per_head, hidden_size)
                sharded_qkv = torch.stack([w_q, w_k, w_v], dim=1)
                sharded_qkv = sharded_qkv.view(num_heads*dims_per_head*3, hidden_size)
                concat_weight[fs_layer + ".attention.query_key_value.weight"] = sharded_qkv
                del concat_weight[hf_layer + ".self_attn.W_pack.weight"]
            else:  
                w_q = concat_weight[hf_layer + ".self_attn.q_proj.weight"].view(num_heads, dims_per_head, hidden_size)
                w_k = concat_weight[hf_layer + ".self_attn.k_proj.weight"].view(num_heads, dims_per_head, hidden_size)
                w_v = concat_weight[hf_layer + ".self_attn.v_proj.weight"].view(num_heads, dims_per_head, hidden_size)
                sharded_qkv = torch.stack([w_q, w_k, w_v], dim=1)
                sharded_qkv = sharded_qkv.view(num_heads*dims_per_head*3, hidden_size)
                
                concat_weight[fs_layer + ".attention.query_key_value.weight"] = sharded_qkv
                
                del concat_weight[hf_layer + ".self_attn.q_proj.weight"]
                del concat_weight[hf_layer + ".self_attn.k_proj.weight"]
                del concat_weight[hf_layer + ".self_attn.v_proj.weight"]
        
        weights_name = _add_variant(WEIGHTS_NAME, None)
        shards, index = shard_checkpoint(concat_weight, weights_name=weights_name)
        
        loaders_map = {}
        weight_map_with_loader = {}
        revert_weight_map = {}
        for k, v in index['weight_map'].items():
            if v in revert_weight_map:
                revert_weight_map[v].append(k)
            else:
                revert_weight_map[v] = [k]
                ld = shards[v]
                loaders_map[v] = ld
            weight_map_with_loader[k] = loaders_map[v]
        
        self.weight_map, self.revert_weight_map, self.loaders = weight_map_with_loader, revert_weight_map, loaders_map.values()
        return fs_config, index
    
    def del_loaded(self, key: str):
        # Remove from memory as we go along
        if key in self.weight_map:
            del self.weight_map[key][key]

    def shard(self, x, dim):
        x_shape = list(x.shape)
        assert x_shape[dim] % self.num_output_shards == 0
        new_x_shape = (
            x_shape[:dim]
            + [self.num_output_shards, x_shape[dim] // self.num_output_shards]
            + x_shape[dim + 1:]
        )
        x = x.view(*new_x_shape)
        return torch.movedim(x, 0, dim)

    def add_sequential_shard(self, dictionary):
        for k, v in dictionary.items():
            for rank in range(self.num_output_shards):
                # self.sequential_cache[rank][f"sequential.{layer_i}.{k}"] = v[rank].clone()
                self.sequential_cache[rank][k] = v[rank].clone()

    def add_sequential_duplicates(self, dictionary):
        for k, v in dictionary.items():
            for rank in range(self.num_output_shards):
                # self.sequential_cache[rank][f"sequential.{layer_i}.{k}"] = v.clone()
                self.sequential_cache[rank][k] = v.clone()

    def add_sequential(self, dictionary, rank):
        for k, v in dictionary.items():
            # self.sequential_cache[rank][f"sequential.{layer_i}.{k}"] = v.clone()
            self.sequential_cache[rank][k] = v.clone()
            
            
def convert_hf_to_fs_mp(input_path:str,
                        output_path:str,
                        model_parallel_size:int=4,
                        is_rm:bool=False,
                        from_hf:bool=True,
                        multiplier:int=1,
                        rm_with_multi_value_head:bool=False,
                        dtype=torch.bfloat16,
                        model_type:Optional[str]=None):
    
    from fengshen_inner.models.llama.modeling_llama import \
        LlamaForCausalLM as FengshenLlama
    if model_type == "baichuan_13B":  # FIXME:修改为枚举类
        hf_config = BaichuanConfig.from_pretrained(input_path)
        fs_config = convert_baichuan_config(hf_config, dtype)
    else:
        hf_config = LlamaConfig.from_pretrained(input_path)
        fs_config = convert_llama_config(hf_config, dtype)
    # mpu.set_model_parallel_world_size(1)
    # mpu.set_model_parallel_rank(0)
    # mpu.set_init_params_in_cuda(False)
    num_heads = hf_config.num_attention_heads
    hidden_size = hf_config.hidden_size
    dims_per_head = hidden_size // num_heads
    
    t1 = time.time()
    logging_rank_0("Convert hf to fs mp...")
    make_output_dir(output_path, model_parallel_size)
    helper = Helper(model_parallel_size=model_parallel_size, input_path=input_path)
    
    # 从HF结构开始转换
    if from_hf:
        config, index = helper.convert_hf_to_fs(hf_config, fs_config, multiplier=multiplier, is_rm=is_rm, rm_with_multi_value_head=rm_with_multi_value_head, is_baichuan=model_type=="baichuan_13B")
    # 从FS结构开始转换
    else:
        config, index = fs_config, helper.weight_map
    num_output_shards = model_parallel_size
    num_heads_per_output_shard = config.num_attention_heads // num_output_shards
    dims_per_head = config.hidden_size // config.num_attention_heads
    for k, v in helper.weight_map.items():
        # embed in and out
        if k in [__HF_EMBED_IN_KEY__, __HF_EMBED_OUT_KEY__]:
            helper.add_sequential_shard({k: helper.shard(v[k], dim=0)})
        elif k.startswith(__HF_NORM_PREFIX__):
            helper.add_sequential_duplicates({k: v[k]})
        
        # value head
        elif k == __HF_VALUE_HEAD_KEY__:
            # helper.add_sequential_duplicates({k: v[k]})
            shard = helper.shard(v[k], dim=1)
            helper.add_sequential_shard({k: shard})
        elif k == __HF_VALUE_HEAD_BIAS_KEY__:
            helper.add_sequential_duplicates({k: v[k]})
        elif k == __HF_TOKEN_VALUE_HEAD_KEY__:
            # helper.add_sequential_duplicates({k: v[k]})
            shard = helper.shard(v[k], dim=1)
            helper.add_sequential_shard({k: shard})
        elif k == __HF_TOKEN_VALUE_HEAD_BIAS_KEY__:
            helper.add_sequential_duplicates({k: v[k]})
        
        elif k.startswith(__HF_LAYER_PREFIX__):
            # QKV weight and bias
            if k.find("query_key_value") != -1:
                output_shape = [num_output_shards, num_heads_per_output_shard *
                                3 * dims_per_head] + list(v[k].shape[1:])
                sharded = v[k].view(output_shape)
                for out_rank in range(num_output_shards):
                    helper.add_sequential({k: sharded[out_rank]}, out_rank)
            # rotary emb
            elif k.find("rotary_emb.inv_freq") != -1:
                helper.add_sequential_duplicates({k: v[k]})
            # layer_norm
            elif k.find("layernorm") != -1:
                helper.add_sequential_duplicates({k: v[k]})
            # linear
            elif k.find("dense") != -1 or k.find("mlp") != -1:
                # 纵切
                if k.find("w2") != -1 or k.find("attention") != -1:
                    if k.find('weight') != -1:
                        shard = helper.shard(v[k], dim=1)
                        helper.add_sequential_shard({k: shard})
                    # bias不切
                    else:
                        helper.add_sequential_duplicates({k: v[k]})
                # 横切
                else:
                    shard = helper.shard(v[k], dim=0)
                    helper.add_sequential_shard({k: shard})
            else:
                logging_rank_0(f"WARNING: unexcept key {k}", level="warning")
        else:
            logging_rank_0(f"WARNING: unexcept key {k}", level="warning")

        helper.del_loaded(k)

    save_splits(index, output_path, helper, config)
    config.save_pretrained(output_path)
    logging_rank_0(f"Use time: {round(time.time() - t1, 3)}")
    return


def launch_convert_fs_mp(config:ModelConvertPipelineConfig):
    
    workspace_path = config.workspace_path
    try:
        if not os.path.exists(workspace_path):
            os.makedirs(workspace_path)
        
        model_path = os.path.join(workspace_path, "models")
        if not os.path.exists(model_path):
            os.mkdir(model_path)
    except:
        pass
    
    if config.reward_model_path:
        rm_fs_mp_path = os.path.join(model_path, "reward_model")
        try:
            if not os.path.exists(rm_fs_mp_path):
                os.mkdir(rm_fs_mp_path)
        except:
            pass
        convert_hf_to_fs_mp(
            input_path=config.reward_model_path,
            output_path=rm_fs_mp_path,
            model_parallel_size=config.tensor_model_parallel_size,
            is_rm=True,
            rm_with_multi_value_head=config.granularity=="token_mix_sample",
            multiplier=config.tensor_model_parallel_size,
            dtype=torch.float16 if config.rm_precision == "fp16" else torch.bfloat16,
            model_type=config.reward_model_type
        )
    
    if config.policy_model_path:
        policy_fs_mp_path = os.path.join(model_path, "policy")
        try:
            if not os.path.exists(policy_fs_mp_path):
                os.mkdir(policy_fs_mp_path)
        except:
            pass
        convert_hf_to_fs_mp(
            input_path=config.policy_model_path,
            output_path=policy_fs_mp_path,
            model_parallel_size=config.tensor_model_parallel_size,
            multiplier=config.tensor_model_parallel_size,
            dtype=torch.float16 if config.policy_precision == "fp16" else torch.bfloat16,
            model_type=config.policy_model_type
        )
    
    return

def check_fs_mp(args:dict):
    # TODO: 检查是否已经有切分好的模型
    return True