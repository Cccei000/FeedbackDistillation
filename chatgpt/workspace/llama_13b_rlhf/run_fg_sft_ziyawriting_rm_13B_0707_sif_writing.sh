#!/bin/sh
#SBATCH --job-name=FG_13B_PPO_13BRM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=100
#SBATCH --gres=gpu:hgx:8 -p pog#-preempted #gpu:hgx:8
#SBATCH --mem-per-cpu=8G # memory per cpu-core (4G is default)
#SBATCH -o ./log/%x-%j.log
#SBATCH -e ./log/%x-%j.err

export NCCL_SOCKET_TIMEOUT=3600
export TORCH_DISTRIBUTED_DETAIL=DEBUG
# export CUDA_LAUNCH_BLOCKING=1

GPUS_PER_NODE=8

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=52299

MODEL_NAME=STEP_PPO_Ziya-Writing-13B-0706_RM13B_writing
# MODEL_NAME=PPO_LLAMA13B_TEST
VERSION="0824_eval_base_t0.8_r1.0"

CODE_PATH='/cognitive_comp/shenjunyi/g/chatgpt/chatgpt/examples'
CODE_NAME="train_llama_ppo_13bSFT_13bRM_step_level.py"

LOG_ROOT_PATH='/cognitive_comp/shenjunyi/g/chatgpt/chatgpt/workspace'
# Prompt数据集地址
# PROMPT_PATH="/cognitive_comp/shenjunyi/g/datasets/prompt/ppo_dataset_math_0626"
PROMPT_PATH="/cognitive_comp/shenjunyi/g/datasets/prompt/ppo_dataset_writing_qa_0718/"
# PROMPT_PATH="/cognitive_comp/songzhuoyang/processed_data/ppopp_dataset_0815_llama2_13b_single/"
# PROMPT_PATH="/cognitive_comp/shenjunyi/g/datasets/prompt/ppo_dataset_writing_eval_100/"

# 经验池保存地址
EXP_SAVE_PATH="${LOG_ROOT_PATH}/exp/$MODEL_NAME/pool/$VERSION/"

# RL过程中Policy保存路径
POLICY_CKPT_PATH="${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/pool/$VERSION/"

# 初始Policy model的路径
# POLICY_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/v2_stage2.2_step6600_MP4"

POLICY_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/Ziya-Writing-13B-0706_fs_MP4"
# POLICY_MODEL_PATH="/cognitive_comp/shenjunyi/g/model/writing/ziya_step_ppo_0815_BoN_global_step2976_fs_MP4"
# POLICY_MODEL_PATH="/cognitive_comp/shenjunyi/g/model/writing/ziya_step_ppo_0815_BoN_global_step4994_fs_MP4"

# POLICY_MODEL_PATH="/cognitive_comp/songzhuoyang/models/llama2_13B_sft/step_26k_fs/"
POLICY_TOKENIZER_PATH="/cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-c-pretrain-tokenizer/"

# RM_MODEL_PATH="/cognitive_comp/liangyuxin/workspace/pipeline/ckpt/reward_model/0817_RM13B_MIX/global_step6517_fs_mp4/"
RM_MODEL_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0725_token/global_step7202_fs_MP4/"
# RM_MODEL_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0803/global_step27458_fs_MP4/"
# RM_MODEL_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0707_ALL/global_step33581_fs_MP4"

# RM_MODEL_PATH="/cognitive_comp/shenjunyi/g/chatgpt/chatgpt/workspace/ckpt/FG_RM_LLAMA_13B_0621_math_zh_en/global_step36570_fs_MP4"
# RM_TOKENIZER_PATH="/cognitive_comp/shenjunyi/g/chatgpt/chatgpt/workspace/ckpt/FG_RM_LLAMA_13B_0621_math_zh_en/global_step36570_fs"

# sif镜像路径
SIF_FILE="/cognitive_comp/liangyuxin/images/fengshen_chatgpt.sif"
CHATGPT_PATH="/cognitive_comp/shenjunyi/g/chatgpt/chatgpt"
FENSHEN_INNER="/cognitive_comp/shenjunyi/g/fengshenbang-lm/fengshen_inner"

EXPERIMENT_ARGS="\
    --wandb_project     $MODEL_NAME \
    --wandb_team        yukawa \
    --wandb_name        "$VERSION-pool" \
    --seed              143 \
    "

PPO_ARGS="\
    --num_episodes      128 \
    --max_timesteps     1 \
    --update_timesteps  1 \
    --sample_batch_size 8 \
    --buffer_limit_size 512 \
    --max_epoch_per_update  2 \
    --eps_clip          0.2 \
    --value_clip        0.2 \
    --gamma             0.99 \
    --lam               0.95 \
    --step_level_reward \
    --enable_reward_scaling \
    --enable_step_ppo \
    --enabling_bon         \
    --min_step_lengths  10 \
    --best_of_n_times   10 \
    "
    # --enable_gae        \

MEGATRON_ARGS="\
    --deepspeed_stage     1 \
    --tensor_model_parallel_size 4 \
    --pipe_model_parallel_size 1 \
    "
    # --offload_optimizer \

EXPERIENCE_ARGS="\
    --top_p                 0.85 \
    --top_k                 0 \
    --repetition_penalty    1. \
    --temperature           0.8 \
    --max_new_tokens        1512 \
    --max_length            2048 \
    --kl_coef               0.0 \
    --experience_batch_size 8 \
    --policy_minibatch_size 8 \
    --rm_minibatch_size     2 \
    --rm_model_max_seq_len  2048 \
    --prompt_dataset_path   /opt/prompt_path/ \
    --exp_save_path         /opt/exp_save_path/ \
    "

MODEL_ARGS="\
    --policy_ckpt_path          /opt/policy_ckpt_path/ \
    --policy_tokenizer_path     /opt/policy_tokenizer_path/ \
    --policy_model_path         /opt/policy_model_path/ \
    --rm_model_path             /opt/rm_model_path/ \
    "
    # --rm_model_tokenizer_path   /opt/rm_tokenizer/ \
    # --rm_model_path             ${RM_MODEL_PATH}/mp_rank_00_model_states.pt \
    # --rm_model_tokenizer_path   ${RM_TOKENIZER_PATH} \
    # --rm_config_path            ${RM_CONFIG_PATH}/config.json \

TRAINER_ARGS="\
    --do_validation     \
    --val_check_interval    5 \
    --val_size_per_task 20 \
    --policy_precision  bf16 \
    --learning_rate     1e-6 \
    --min_learning_rate 1e-7 \
    --total_steps       512 \
    --warmup_ratio      0.01 \
    --policy_train_batch_size   1 \
    --scheduler_type    cosine \
    --actor_lr  1e-6 \
    --critic_lr 1e-6 \
    "
    # 

export options=" \
    $EXPERIMENT_ARGS \
    $MEGATRON_ARGS \
    $PPO_ARGS \
    $EXPERIENCE_ARGS \
    $MODEL_ARGS \
    $TRAINER_ARGS \
    "

export CMD="${CODE_PATH}/${CODE_NAME} $options"
echo "START"

# mkdir ${LOG_ROOT_PATH}/exp/$MODEL_NAME/
# mkdir ${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/
# mkdir ${LOG_ROOT_PATH}/exp/$MODEL_NAME/pool/
# mkdir ${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/pool/
mkdir -p ${LOG_ROOT_PATH}/exp/$MODEL_NAME/pool/$VERSION/
mkdir -p ${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/pool/$VERSION/

# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_TIMEOUT=3600
# export TORCH_DISTRIBUTED_DETAIL=DEBUG
# export NCCL_P2P_LEVEL=NVL
singularity exec --nv \
    -B $PROMPT_PATH:/opt/prompt_path/ \
    -B $EXP_SAVE_PATH:/opt/exp_save_path/ \
    -B $POLICY_CKPT_PATH:/opt/policy_ckpt_path/ \
    -B $POLICY_TOKENIZER_PATH:/opt/policy_tokenizer_path/ \
    -B $POLICY_MODEL_PATH:/opt/policy_model_path/ \
    -B $RM_MODEL_PATH:/opt/rm_model_path/ \
    -B $CODE_PATH:/opt/code/ \
    -B $CHATGPT_PATH:/opt/chatgpt/chatgpt/ \
    -B $FENSHEN_INNER:/opt/fengshenbang-lm/fengshen_inner/ \
    $SIF_FILE python3 -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE /opt/code/$CODE_NAME $options 

set +x

    # -B $RM_TOKENIZER_PATH:/opt/rm_tokenizer/ \
