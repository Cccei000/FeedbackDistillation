# encoding=utf-8
import os
import warnings
from glob import glob
import gc
import argparse
import datasets as ds
import pandas as pd
import torch
import torch.distributed.run
import deepspeed
from fengshen_inner.models.llama.modeling_llama import LlamaModel
from fengshen_inner.models.megatron import mpu
from fengshen_inner.models.model_utils import (add_inverse_square_args,
                                               add_module_args)
from transformers import LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from chatgpt.pipeline import launch_ppo
from chatgpt.pipeline.config import ActorGranularity, RewardModelGranularity

from chatgpt.dataset import RMCollator
from chatgpt.nn.llama import LlamaRM
from chatgpt.replay_buffer import DistributedBatchSampler
from chatgpt.strategies import (add_megatron_deepspeed_args,
                                build_deepspeed_config, initialize_megatron,
                                setup_model_and_optimizer)
from chatgpt.nn.utils import masked_mean
from chatgpt.utils import is_rank_0,print_rank_0

def add_neox_ppo_pipeline_args(parent_args:argparse.ArgumentParser):
    group = parent_args.add_argument_group("Data Args")
    group.add_argument("--dataset_path", type=str, default= None)

    group = parent_args.add_argument_group("Trainer Args")
    group.add_argument("--num_workers", type=int, default=2)
    group.add_argument("--rm_batch_size", type=int, default=1)
    group.add_argument("--rm_ckpt_path", type=str, default=None)

    group = parent_args.add_argument_group("Model Args")
    group.add_argument("--rm_tokenizer_path", type=str, default=None)
    group.add_argument("--workspace_path", type=str, default=None)
    group.add_argument("--rm_model_path", type=str, default=None)
    group.add_argument("--rm_max_seq_len", type=int, default=1024)
    group.add_argument("--granularity", type=str, default=None)

 
    return parent_args

def caculate_loss_and_acc(output,pair_length,granularity,action_mask=None):
    losses = []
    pos_rewards_list = []
    neg_rewards_list = []

    if granularity=="sample":
        rewards = output
    elif granularity=="token":
        rewards = masked_mean(output,action_mask)
    else:
        raise Exception("granularity not supported")

    for start, end in zip(pair_length[:-1], pair_length[1:]):
        pairs = torch.combinations(torch.arange(end - start, device=output.device), 2)
        _loss,pos_rewards,neg_rewards = caculate_loss(start,pairs,rewards,output)
        losses.append(_loss)
        pos_rewards_list.append(pos_rewards)
        neg_rewards_list.append(neg_rewards)
    loss = torch.stack(losses).mean()

    return pos_rewards_list, neg_rewards_list, loss 

def caculate_loss(start,pairs,rewards,output):
    loss_func = torch.nn.Sigmoid()
    pos_ids, neg_ids = pairs[:, 0], pairs[:, 1]
    pos_rewards = rewards.take(start + pos_ids)
    neg_rewards = rewards.take(start + neg_ids)
    pos = output[start + pos_ids]
    neg = output[start + neg_ids]
    l2 = 0.5*(pos**2+neg**2).mean()
    loss = (-torch.log(loss_func(pos_rewards - neg_rewards)) + 0.01 * l2).mean()
    return loss,pos_rewards,neg_rewards

def test(args):

    initialize_megatron(args)
    strategy = build_deepspeed_config(args)
    special_token_dict = {'pad_token': '</s>'}
    tokenizer = LlamaTokenizer.from_pretrained(args.rm_tokenizer_path, use_fast=False)
    tokenizer.add_special_tokens(special_token_dict)
    args.train_micro_batch_size_per_gpu=args.rm_batch_size

    ##### Load dataset #####
    collate_fn = RMCollator(tokenizer=tokenizer, max_length=args.rm_max_seq_len, query_key="query", response_key="preference")
    data_path = os.path.join(args.dataset_path,"test.jsonl")
    df = pd.read_json(data_path, lines=True)
    dataset = ds.Dataset.from_pandas(df)

    # load reward model
    rm_precision = torch.bfloat16 #if args.rm_precision == "bf16" else torch.float16 
    sft_model_path =  os.path.join(args.workspace_path, "models/policy")
    model = LlamaModel.from_pretrained(
        os.path.join(sft_model_path,f"part_{mpu.get_model_parallel_rank()}"),
        torch_dtype=torch.bfloat16
    )
    # model.is_bidirectional = True
    reward_model = LlamaRM(
        model=model,
        value_head=None,
        rm_granularity = RewardModelGranularity[args.granularity],
        output_granularity = ActorGranularity[args.granularity],
    ).to(dtype=rm_precision)
    reward_model.load_state_dict(torch.load(os.path.join(args.rm_model_path, f"mp_rank_{str(mpu.get_model_parallel_rank()).zfill(2)}_model_states.pt"))['module'])
    print(reward_model.value_head.weight)

    infer_config = {
        "tensor_parallel": {"tp_size": args.tensor_model_parallel_size},
        "dtype": "bf16",
        "enable_cuda_graph": False,
        "injection_policy":{LlamaDecoderLayer: ('self_attn.o_proj', 'mlp.down_proj')}
    }
    reward_model = deepspeed.init_inference(reward_model,config=infer_config)
    reward_model.eval()

    rank = mpu.get_data_parallel_rank()
    world_size = mpu.get_data_parallel_world_size()
    global_batch_size = args.train_micro_batch_size_per_gpu * world_size
    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(
        sampler=sampler,
        batch_size=global_batch_size,
        drop_last=True,
        rank=rank,
        world_size=world_size,
    )

    # Torch dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
    )

    device = torch.cuda.current_device()
    all_rewards = []
    with torch.no_grad():
        for batch in dataloader:
            rewards = reward_model(batch['input_ids'].to(device), attention_mask = batch['attention_mask'].to(device))
            if args.granularity=="token":
                rewards = rewards* batch["action_mask"].to(model.device)
            all_rewards.append(rewards.cpu().tolist())
            gc.collect()
            torch.cuda.empty_cache()
        df["reward"] = all_rewards
        df.to_json(data_path.replace(".jsonl","_scored.jsonl"),orient="records",lines=True,force_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_module_args(parent_args=parser)
    parser = add_inverse_square_args(parent_args=parser)
    parser = add_neox_ppo_pipeline_args(parent_args=parser)
    parser = add_megatron_deepspeed_args(parent_args=parser)
    args = parser.parse_args()
    print(args)
    test(args)
    print("test finished!")
