#!/bin/sh
#SBATCH --job-name=PPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:hgx:4 -p pog # -preempted
#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G is default)
#SBATCH -o ./%x-%j.log
#SBATCH -e ./%x-%j.err

GPUS_PER_NODE=4

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=52299

MODEL_NAME=PPO_LLAMA13B_V2_Merged_average_chat_math_RM_13B_0713_token_level
VERSION="0717_token_level"

CODE_PATH='/cognitive_comp/liangyuxin/chatgpt/chatgpt/examples'
CODE_NAME="train_llama_ppo_13bSFT_13bRM_token_level.py"

LOG_ROOT_PATH='/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RLHF'
# Prompt数据集地址
PROMPT_PATH="/cognitive_comp/liangyuxin/datasets/RL/prompt_dataset_no_math_0627_mix"

# 经验池保存地址
EXP_SAVE_PATH="${LOG_ROOT_PATH}/exp/$MODEL_NAME/pool/$VERSION/"

# RL过程中Policy保存路径
POLICY_CKPT_PATH="${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/pool/$VERSION/"

# 初始Policy model的路径
POLICY_MODEL_PATH="/cognitive_comp/liangyuxin/models/SFT/13B/merged_average-chat_19000_18000-math_6000_fs_MP4"
POLICY_TOKENIZER_PATH="/cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-c-pretrain-tokenizer/"
RM_MODEL_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0707_token/global_step35262_fs_MP4"
RM_TOKENIZER_PATH="/cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-c-pretrain-tokenizer/"

# sif镜像路径
SIF_FILE="/cognitive_comp/liangyuxin/images/fengshen_chatgpt.sif"
CHATGPT_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt"
FENSHEN_INNER="/cognitive_comp/liangyuxin/fengshen_inner/fengshenbang-lm/fengshen_inner"

EXPERIMENT_ARGS="\
    --wandb_project     $MODEL_NAME \
    --wandb_team        yukawa \
    --wandb_name        "$VERSION-pool" \
    --seed              42 \
    "

PPO_ARGS="\
    --num_episodes      256 \
    --max_timesteps     1 \
    --update_timesteps  1 \
    --sample_batch_size 128 \
    --buffer_limit_size 512 \
    --max_epoch_per_update  2 \
    --eps_clip          0.2 \
    --value_clip        200 \
    --gamma             1.0 \
    --lam               0.95 \
    --enable_gae       \
    --token_level_reward    \
    "

MEGATRON_ARGS="\
    --deepspeed_stage     1 \
    --tensor_model_parallel_size 4 \
    --pipe_model_parallel_size 1 \
    --offload_optimizer \
    "

EXPERIENCE_ARGS="\
    --top_p                 0.85 \
    --top_k                 0 \
    --repetition_penalty    1. \
    --temperature           0.8 \
    --max_new_tokens        512 \
    --max_length            2048 \
    --kl_coef               0.05\
    --experience_batch_size 128 \
    --gen_minibatch_size    32 \
    --policy_minibatch_size 8 \
    --rm_minibatch_size     1 \
    --rm_model_max_seq_len  2048 \
    --prompt_dataset_path   /opt/prompt_path/ \
    --exp_save_path         /opt/exp_save_path/ \
    "

MODEL_ARGS="\
    --policy_ckpt_path          /opt/policy_ckpt_path/ \
    --policy_tokenizer_path     /opt/policy_tokenizer_path/ \
    --policy_model_path         /opt/policy_model_path/ \
    --rm_model_path             /opt/rm_model_path/ \
    --rm_model_tokenizer_path   /opt/rm_tokenizer/ \
    "

TRAINER_ARGS="\
    --do_validation     \
    --val_check_interval 5 \
    --val_size_per_task  10 \
    --policy_precision  bf16 \
    --learning_rate     1e-6 \
    --min_learning_rate 1e-7 \
    --total_steps       512 \
    --warmup_ratio      0.01 \
    --policy_train_batch_size   1 \
    --scheduler_type    cosine \
    --entropy_loss_coef 0.01 \
    --entropy_loss_decay_rate 1.0 \
    "
    # --clip_grad         \


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

mkdir -p ${LOG_ROOT_PATH}/exp/$MODEL_NAME/pool/$VERSION/
mkdir -p ${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/pool/$VERSION/

singularity exec --nv \
    -B $PROMPT_PATH:/opt/prompt_path/ \
    -B $EXP_SAVE_PATH:/opt/exp_save_path/ \
    -B $POLICY_CKPT_PATH:/opt/policy_ckpt_path/ \
    -B $POLICY_TOKENIZER_PATH:/opt/policy_tokenizer_path/ \
    -B $POLICY_MODEL_PATH:/opt/policy_model_path/ \
    -B $RM_MODEL_PATH:/opt/rm_model_path/ \
    -B $RM_TOKENIZER_PATH:/opt/rm_tokenizer/ \
    -B $CODE_PATH:/opt/code/ \
    -B $CHATGPT_PATH:/opt/chatgpt/chatgpt/ \
    -B $FENSHEN_INNER:/opt/fengshenbang-lm/fengshen_inner/ \
    $SIF_FILE python3 -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE /opt/code/$CODE_NAME $options

set +x
