# encoding=utf-8

"""Train"""
import os
import torch
import torch.distributed.run
import torch.nn.functional as F
import argparse
import datasets as ds

from transformers.models.llama import LlamaForCausalLM as HFLlamaForCausalLM
from fengshen_inner.models.model_utils import add_module_args, add_inverse_square_args
from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM
from fengshen_inner.models.megatron import mpu

from chatgpt.dataset import ReplayBufferDataset
from chatgpt.nn.llama import LlamaActor, modeling_fengshenLlama_rm
from chatgpt.experience_maker import InferenceExperienceMaker
from chatgpt.replay_buffer import DistributedBatchSampler
from chatgpt.utils import local_rank, print_rank_0
from chatgpt.strategies import add_megatron_deepspeed_args, initialize_megatron, build_deepspeed_config, setup_model_and_optimizer, get_save_checkpoint_callback
from chatgpt.nn.utils import zero_pad_sequences
from transformers import LlamaTokenizer, AutoTokenizer
from tokenizers import AddedToken
from chatgpt.pipeline.tokenizer import llama2_to_llama

_POLICY_TOKENIZER_PATH = "/cognitive_comp/songzhuoyang/models/llama_sft/20230405v1"
_PRETRAIN_MODEL_PATH = '/cognitive_comp/wanghao/models/llama_sft/13b_0423_MP2'
_REWARD_MODEL_PATH = '/cognitive_comp/liangyuxin/workspace/rm_train/RM_0412_mix_7B/ckpt/last.ckpt/checkpoint/mp_rank_00_model_states.pt'
_REWARD_CONFIG_PATH = '/cognitive_comp/sunqianguo/pretrained/checkpoints/7B/0405/v2/checkpoint-16000/config.json'
_PPO_DATASET_PATH = '/cognitive_comp/songzhuoyang/processed_data/mixed_ppo_dataset_0327_for_llama'


_SPECIAL_TOKENS_DICT = {'pad_token': '</s>'}
human = AddedToken("<human>",lstrip=False,rstrip=False,single_word=False,normalized=True)
bot = AddedToken("<bot>",lstrip=False,rstrip=False,single_word=False,normalized=True)

def add_neox_ppo_pipeline_args(parent_args:argparse.ArgumentParser):
    
    group = parent_args.add_argument_group("Experiment Args")
    
    group.add_argument("--wandb_project", type=str, default="PPO_LLAMA")
    group.add_argument("--wandb_group", type=str, default="")
    group.add_argument("--wandb_team", type=str, default=None)
    group.add_argument("--wandb_name", type=str, default=None)
     
    group = parent_args.add_argument_group("PPO Args")
    group.add_argument("--num_episodes", type=int, default=1, help="训练轮数，每轮中包括经验池采样、奖励模型打分、模型训练三步")
    group.add_argument("--max_timesteps", type=int, default=1, help="每轮中进行经验池采样的次数")
    group.add_argument("--update_timesteps", type=int, default=1, help="")
    group.add_argument("--sample_replay_buffer", action="store_true", default=False)
    group.add_argument("--sample_batch_size", type=int, default=32, help="每次经验池采样中，使用的Prompt数量（不考虑数据并行）")
    group.add_argument("--buffer_limit_size", type=int, default=512)
    group.add_argument("--max_epoch_per_update", type=int, default=2, help="每次模型训练时，训练的epoch数")
    group.add_argument("--replay_buffer_cpu_offload", type=bool, default=True)
    group.add_argument("--entropy_loss_coef", type=float, default=0.01)
    group.add_argument("--entropy_loss_decay_rate", type=float, default=0.98)
    group.add_argument("--clip_grad", action="store_true", default=False)
    
    group.add_argument("--eps_clip", type=float, default=0.2)
    group.add_argument("--value_clip", type=float, default=0.2)
    
    group.add_argument("--enable_gae", action="store_true", default=False)
    group.add_argument("--gamma", type=float, default=1.0)
    group.add_argument("--lam", type=float, default=0.95)
    
    group = parent_args.add_argument_group("Experience Args")
    group.add_argument("--top_p", type=float, default=0.85)
    group.add_argument("--top_k", type=int, default=0)
    group.add_argument("--kl_coef", type=float, default=0.0)
    group.add_argument("--max_length", type=int, default=1024)
    group.add_argument("--max_new_tokens", type=int, default=512)
    group.add_argument("--repetition_penalty", type=float, default=1.)
    group.add_argument("--temperature", type=float, default=1.)
    group.add_argument("--experience_batch_size", type=int, default=32)
    group.add_argument("--policy_minibatch_size", type=int, default=4)
    group.add_argument("--gen_minibatch_size", type=int, default=4)
    group.add_argument("--rm_minibatch_size", type=int, default=1)
    group.add_argument("--rm_max_seq_len", type=int, default=2048)
    group.add_argument("--prompt_dataset_path", type=str, default=_PPO_DATASET_PATH, help="用于训练的所有prompt") # 格式参考默认路径的dataset
    group.add_argument("--exp_save_path", type=str, default="/cognitive_comp/songzhuoyang/workspace/chatgpt/6B_rlhf/exp", help="训练产生的经验池的保存路径") 
    
    group = parent_args.add_argument_group("Trainer Args")
    group.add_argument("--num_workers", type=int, default=2)
    group.add_argument("--total_steps", type=int, default=1e4)
    group.add_argument("--policy_train_batch_size", type=int, default=1)
    group.add_argument("--do_validation", action="store_true", default=False)
    group.add_argument("--val_check_interval", type=int, default=5)
    group.add_argument("--val_size_per_task", type=int, default=20)
    
    group = parent_args.add_argument_group("Model Args")
    group.add_argument("--policy_ckpt_path", type=str, default="/cognitive_comp/songzhuoyang/workspace/chatgpt/7B_rlhf/ckpt", help="rlhf ckpt保存的根目录") # 训练过程中会自动创建文件夹，保存每个episode的ckpt
    group.add_argument("--policy_tokenizer_path", type=str, default=_POLICY_TOKENIZER_PATH) # tokenizer路径
    group.add_argument("--policy_model_path", type=str, default=_PRETRAIN_MODEL_PATH, help="生成模型sft ckpt路径") # 需要使用gpt-neox/utils中的gxy写的转换脚本转换参数的key
    group.add_argument("--rm_config_path", type=str, default=_REWARD_CONFIG_PATH, help="rm config路径") # 训练过程中会自动创建文件夹，保存每个episode的ckpt
    group.add_argument("--rm_model_path", type=str, default=_REWARD_MODEL_PATH, help="rm ckpt路径") # 需要使用gpt-neox/utils中的gxy写的转换脚本转换参数的key
    group.add_argument("--rm_model_tokenizer_path", type=str, default=_POLICY_TOKENIZER_PATH) # tokenizer路径
    # group.add_argument("--ref_model_path", type=str, default=_PRETRAIN_MODEL_PATH, help="reference ckpt路径") # 需要使用gpt-neox/utils中的gxy写的转换脚本转换参数的key

    return parent_args
   

def load_dateset(args):
    
    import datasets as ds
    ds.disable_caching()
    dataset = ds.load_from_disk(args.prompt_dataset_path)
    dataset = dataset.train_test_split(train_size=1, shuffle=False)["train"]
    prompts = list(dataset["query"])
    
    queries = [f"<Human Round-1>:{item}\n<Assistant Round-1>:" for item in prompts]

    return prompts, queries

def launch(args):
    """Main training program.

    """
    
    ds.disable_caching()

    initialize_megatron(args)
    
    strategy = build_deepspeed_config(args)

    # llama_tokenizer = LlamaTokenizer.from_pretrained(args.policy_tokenizer_path)
    llama_tokenizer = AutoTokenizer.from_pretrained(args.policy_tokenizer_path)
    llama_tokenizer.add_special_tokens(_SPECIAL_TOKENS_DICT)
    llama_tokenizer.add_special_tokens({"additional_special_tokens":[human, bot]})

    # deepspeed actor & critic
    # state_dict = torch.load(
    #     f"{args.policy_ckpt_path}/mp_rank_{str(mpu.get_model_parallel_rank()).zfill(2)}_model_states.pt",
    #     map_location="cpu"
    # )["module"]
    # new_state_dict = {
    #     key[len("model."):]: value for key, value in state_dict.items()
    # }
    
    print("Load ckpt...")
    llama_model = LlamaForCausalLM.from_pretrained(
        f"{args.policy_model_path}/part_{mpu.get_model_parallel_rank()}",
        torch_dtype=torch.float16
    ).cuda()
    # llama_model.load_state_dict(new_state_dict, strict=True)
    
    actor = LlamaActor(
        model=llama_model
    ).to(dtype=torch.float16)
    
    actor, actor_optimizer, actor_lr = setup_model_and_optimizer(args, actor, strategy)

    # hf_path = "/cognitive_comp/zhangwenjun/checkpoints/llama-neox-sft/13B-llama2-20230809_stage2/global_step26000-hf"
    hf_path = "/cognitive_comp/zhangwenjun/checkpoints/llama-neox-sft/merged_0630/merged_average-chat_19000-mmm_0615_ind_chat_19000_math_6000-mer_0619_ind_chat_19000_18000_math_6000"
    hf_model = HFLlamaForCausalLM.from_pretrained(hf_path)
    hf_model = hf_model.to(dtype=torch.float16, device=actor.model.device)
    hf_model = hf_model.eval().cuda()

    prompts, queries = load_dateset(args)
    
        
    print("Validation")
    encode_dict = llama_tokenizer.batch_encode_plus(queries)
    input_ids = [[ids] for ids in encode_dict["input_ids"]] # 转换为torch.Tensor
    attention_mask = [[msk] for msk in encode_dict["attention_mask"]]
    
    hf_output_logits = []
    fs_output_logits = []
    
    actor.eval()
    with torch.no_grad():
        for ids, msk in zip(input_ids, attention_mask):
            ids = torch.tensor(ids, device=actor.model.device)
            msk = torch.tensor(msk, device=actor.model.device)
            fs_output = actor.module.model(
                ids, attention_mask=msk, use_cache=True, output_hidden_states=True, output_attentions=True,
            )
            hf_output = hf_model(
                ids, attention_mask=msk, output_hidden_states=True, use_cache=True, output_attentions=True,
            )
            # fs_logits = fs_output[0]
            # hf_logits = hf_output.logits
            # fs_log_probs = F.log_softmax(fs_logits, dim=-1)
            # hf_log_progs = F.log_softmax(hf_logits, dim=-1)
            fs_output_logits.append(fs_output)
            hf_output_logits.append(hf_output)
            
    
    
    # hf_dataset = ds.Dataset.from_dict(
    #     {
    #         "query": queries,
    #         "input_ids": input_ids,
    #         "attn_msk": attention_mask,
    #         "fs_logprobs": fs_output_logits,
    #         "hf_logprobs": hf_output_logits
    #     }
    # )
    
    mp_rank = mpu.get_model_parallel_rank()
    pp_rank = mpu.get_pipe_parallel_rank()
    if mp_rank == 0 and pp_rank == 0:
        # hf_dataset.save_to_disk(os.path.join(args.exp_save_path))
        fs_logits_dict = dict(zip(range(len(fs_output_logits)), fs_output_logits))
        hf_logits_dict = dict(zip(range(len(hf_output_logits)), hf_output_logits))
        torch.save(fs_logits_dict, os.path.join(args.exp_save_path, "fs_hd_state.pt"))
        torch.save(hf_logits_dict, os.path.join(args.exp_save_path, "hf_hd_state.pt"))

    return


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser = add_module_args(parent_args=parser)
    parser = add_inverse_square_args(parent_args=parser)
    parser = add_neox_ppo_pipeline_args(parent_args=parser)
    parser = add_megatron_deepspeed_args(parent_args=parser)
    
    args = parser.parse_args()
    
    launch(args)
    print("!")
