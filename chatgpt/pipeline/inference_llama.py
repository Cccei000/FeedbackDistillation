# encoding=utf-8

"""Train"""
import argparse
import os

import datasets as ds
import torch
import torch.distributed.run
import torch.nn.functional as F
from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM
from fengshen_inner.models.megatron import mpu
from tokenizers import AddedToken
from tqdm import tqdm
from transformers import AutoTokenizer

from chatgpt.nn.llama import LlamaActor
from chatgpt.nn.utils import zero_pad_sequences
from chatgpt.strategies import (build_deepspeed_config, initialize_megatron,
                                setup_model_and_optimizer)
from chatgpt.utils import print_rank_0
from chatgpt.pipeline.config import ActorGranularity, RewardModelGranularity
from chatgpt.models.baichuan import BaichuanTokenizer

_SPECIAL_TOKENS_DICT = {'pad_token': '</s>'}
human = AddedToken("<human>",lstrip=False,rstrip=False,single_word=False,normalized=True)
bot = AddedToken("<bot>",lstrip=False,rstrip=False,single_word=False,normalized=True)

def load_dateset(args):
    
    import datasets as ds
    ds.disable_caching()
    dataset = ds.load_from_disk(args.prompt_dataset_path)
    # dataset = dataset.train_test_split(train_size=16, shuffle=False)["train"]
    prompts = list(dataset["query"])
    
    # queries = [f"<Human Round-1>:{item}\n<Assistant Round-1>:" for item in prompts]
    # queries = [f"<human>:{item}\n<bot>:" for item in prompts]
    queries = [f"<reserved_106>{item}<reserved_107>" for item in prompts]

    return prompts, queries

def launch_inference(args):
    """Main training program.

    """
    
    ds.disable_caching()

    initialize_megatron(args)
    
    strategy = build_deepspeed_config(args)

    # llama_tokenizer = LlamaTokenizer.from_pretrained(args.policy_tokenizer_path)
    llama_tokenizer = BaichuanTokenizer.from_pretrained(args.policy_tokenizer_path)
    # llama_tokenizer.add_special_tokens(_SPECIAL_TOKENS_DICT)
    # llama_tokenizer.add_special_tokens({"additional_special_tokens":[human, bot]})
    
    # reward model
    # reward_model = modeling_fengshenLlama_rm(
    #     pretrained_path=f"{args.rm_model_path}/part_{mpu.get_model_parallel_rank()}",\
    #     convert_func=llama2_to_llama(args, llama_tokenizer, llama_tokenizer)
    # ).eval().half().cpu()

    # deepspeed actor & critic
    state_dict = torch.load(
        f"{args.policy_ckpt_path}/mp_rank_{str(mpu.get_model_parallel_rank()).zfill(2)}_model_states.pt",
        map_location="cpu"
    )["module"]
    new_state_dict = {
        key[len("model."):]: value for key, value in state_dict.items()
    }
    
    print("Load ckpt...")
    llama_model = LlamaForCausalLM.from_pretrained(
        f"{args.policy_model_path}/part_{mpu.get_model_parallel_rank()}",
        torch_dtype=torch.float16
    ).cuda()
    llama_model.load_state_dict(new_state_dict, strict=True)
    
    actor = LlamaActor(
        model=llama_model,
        actor_granularity=ActorGranularity.sample,
        rm_granularity=RewardModelGranularity.sample
    ).to(dtype=torch.float16)
    
    # hf_path = "/cognitive_comp/zhangwenjun/checkpoints/llama-neox-sft/merged_0630/merged_average-chat_19000-mmm_0615_ind_chat_19000_math_6000-mer_0619_ind_chat_19000_18000_math_6000"
    # hf_model = HFLlamaForCausalLM.from_pretrained(hf_path)
    # hf_model = hf_model.to(dtype=torch.float16, device=actor.model.device)
    # hf_model = hf_model.eval().cuda()
    
    tokenizer_vocab_size = llama_tokenizer.vocab_size
    policy_vocab_size = actor.model.config.vocab_size
    
    actor, actor_optimizer, actor_lr = setup_model_and_optimizer(args, actor, strategy)

    # 初始化experience_maker replay_buffer

    # experience_maker = InferenceExperienceMaker(
    #     actor=actor,
    #     reward_model=reward_model,
    #     seed=args.seed + mpu.get_data_parallel_rank(),
    #     pad_token_id=llama_tokenizer.pad_token_id,
    #     gen_minibatch_size=args.gen_minibatch_size,
    #     rm_minibatch_size=args.rm_minibatch_size,
    #     enable_gae=args.enable_gae
    # )

    bad_words_ids = [[llama_tokenizer.convert_tokens_to_ids(role)] for role in ['<human>','<bot>']]
    print_rank_0(f"Ignore all role tokens: bad_words_ids")

    if policy_vocab_size > tokenizer_vocab_size:
        bad_words_ids += [[ids] for ids in range(tokenizer_vocab_size, policy_vocab_size)]
        print_rank_0(f"BAD TOKEN IDS: {tokenizer_vocab_size}~{policy_vocab_size - 1}")

    # 初始化trainer
    generate_kwargs = {
        "do_sample": True,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "bad_words_ids": bad_words_ids,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "use_cache": True,
        "pad_token_id": llama_tokenizer.pad_token_id,
        "eos_token_id": llama_tokenizer.eos_token_id,
    }

    prompts, queries = load_dateset(args)
    
        
    print("Validation")
    inputs = llama_tokenizer.batch_encode_plus(queries)["input_ids"]
    input_ids = [torch.tensor(ids) for ids in inputs] # 转换为torch.Tensor
    input_ids = zero_pad_sequences(input_ids, side="right", # side="left",
                                padding_value=generate_kwargs["pad_token_id"])
    sequences = []
    # hf_sequences = []
    
    actor.eval()
    with torch.no_grad():
        for i in tqdm(range(0, input_ids.shape[0], args.gen_minibatch_size)):
            mini_batch_input = input_ids[i: i + args.gen_minibatch_size]
            mini_batch_sequences, _, _, _ = actor.module.generate(mini_batch_input.cuda(),**generate_kwargs)
            sequences.append(mini_batch_sequences.cpu())
            
            # mini_batch_sequences_hf = hf_model.generate(mini_batch_input.cuda(), **generate_kwargs)
            # hf_sequences.append(mini_batch_sequences_hf.cpu())
           
    # FS 
    max_len = max(item.shape[1] for item in sequences)
    padded_sequences = [
        F.pad(item, (0, max_len - item.shape[1]), value=llama_tokenizer.pad_token_id) # 使用传入的pad_token_id填充
        for item in sequences
    ]
    padded_sequences = torch.cat(padded_sequences, dim=0)
    sequences = padded_sequences.tolist()
    
    # HF
    # max_len = max(item.shape[1] for item in hf_sequences)
    # padded_sequences = [
    #     F.pad(item, (0, max_len - item.shape[1]), value=llama_tokenizer.pad_token_id) # 使用传入的pad_token_id填充
    #     for item in hf_sequences
    # ]
    # padded_sequences = torch.cat(padded_sequences, dim=0)
    # hf_sequences = padded_sequences.tolist()
    
    responses = []
    # hf_responses = []
    for q, r_ids in zip(queries, sequences):
        # text = llama_tokenizer.decode(r_ids, skip_special_tokens=False)
        # text = text.replace('<s>','').replace('</s>', '')[len(q) + 3:]
        text = llama_tokenizer.decode(r_ids, skip_special_tokens=True)
        # hf_text = llama_tokenizer.decode(hf_ids, skip_special_tokens=True)
        responses.append(text)
        # hf_responses.append(hf_text)
    
    hf_dataset = ds.Dataset.from_dict({
        "query": prompts,
        "response": responses,
        # "hf_response": hf_responses,
    })

    mp_rank = mpu.get_model_parallel_rank()
    pp_rank = mpu.get_pipe_parallel_rank()
    if mp_rank == 0 and pp_rank == 0:
        hf_dataset.save_to_disk(os.path.join(args.exp_save_path))

    return


