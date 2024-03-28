#!/bin/bash
#
#SBATCH --job-name=chatgpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=20 #
#SBATCH --gres=gpu:8                 # number of gpus
##SBATCH -x dgx047
##SBATCH --reservation=acagpt
#SBATCH -o outputs/%x-%j.log
#SBATCH -e outputs/%x-%j.err
# SBATCH --requeue
# SBATCH --qos=preemptive

GPUS_PER_NODE=8

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=52299

MODEL_NAME=PPO_LLAMA13B_V2_S2_S2.2_S3.1_RM65B0602
VERSION="0608_1500"

CODE_PATH='/cognitive_comp/wanghao/experiments/chatgpt/chatgpt/examples'
CODE_NAME="train_llama_ppo_13bSFT_65bRM.py"

LOG_ROOT_PATH='/cognitive_comp/wanghao/experiments/workspace/chatgpt/llama_13b_rlhf'
# Prompt数据集地址

# PROMPT_PATH="/cognitive_comp/wanghao/data/processed_data/mixed_ppo_dataset_0507_for_llama_13b"
# PROMPT_PATH="/cognitive_comp/wanghao/data/processed_data/mixed_ppo_dataset_0524_for_llama_13b"
# PROMPT_PATH="/cognitive_comp/wanghao/data/processed_data/mixed_ppo_dataset_0529_for_llama_13b"
# PROMPT_PATH="/cognitive_comp/wanghao/data/processed_data/mixed_ppo_dataset_0518_for_llama_13b"
# PROMPT_PATH="/cognitive_comp/songzhuoyang/processed_data/mixed_ppo_dataset_0423_for_llama_13b/"
PROMPT_PATH="/cognitive_comp/wanghao/data/processed_data/mixed_ppo_dataset_0601_for_llama_13b"
# PROMPT_PATH="/cognitive_comp/wanghao/data/processed_data/mixed_ppo_dataset_0602_dialog_for_llama_13b"

# 经验池保存地址
EXP_SAVE_PATH="${LOG_ROOT_PATH}/exp/$MODEL_NAME/pool/$VERSION/"

# RL过程中Policy保存路径
POLICY_CKPT_PATH="${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/pool/$VERSION/"

# 初始Policy model的路径
# POLICY_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/v2_stage2.2_step6600_MP4"
# POLICY_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/v2_stage3_step10800_MP4"
POLICY_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/merged_stage2.2step6600_stage3step11800_stage3.1step7000_MP8"
# POLICY_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/llama_13B_v2_s2.2_s3_s3.1_rlhf_0601_RM13B0525_step44_MP4"
POLICY_TOKENIZER_PATH="/cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-c-pretrain-tokenizer/"

# RM_CONFIG_PATH="/cognitive_comp/sunqianguo/pretrained/checkpoints/7B/0405/v2/checkpoint-16000/"
# RM_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/rm_13b_0510_MP4"
# RM_MODEL_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0520/global_step55666_fs"
# RM_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/rm_13b_0525_MP4"
RM_MODEL_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/65B_RM/ckpt/RM_LLAMA_65B_0602/global_step57241_fs"
# RM_TOKENIZER_PATH="/cognitive_comp/songzhuoyang/models/llama_sft/20230405v1/"
# RM_CONFIG_PATH="/cognitive_comp/wanghao/models/reward_model"
# RM_MODEL_PATH="/cognitive_comp/wanghao/models/reward_model"
# RM_TOKENIZER_PATH="/cognitive_comp/wanghao/models/reward_model"

# sif镜像路径
# SIF_FILE="/cognitive_comp/songzhuoyang/images/fengshen_chatgpt.sif"


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
    --value_clip        0.2 \
    --gamma             1.0 \
    --lam               0.95 \
    "
    # --enable_gae        \

MEGATRON_ARGS="\
    --deepspeed_stage     1 \
    --tensor_model_parallel_size 8 \
    --pipe_model_parallel_size 1 \
    "
    # --offload_optimizer \

EXPERIENCE_ARGS="\
    --top_p                 0.85 \
    --top_k                 0 \
    --repetition_penalty    1. \
    --temperature           0.8 \
    --max_length            1900 \
    --kl_coef               0 \
    --experience_batch_size 128 \
    --policy_minibatch_size 8 \
    --rm_minibatch_size     1 \
    --rm_model_max_seq_len  1900 \
    --prompt_dataset_path   ${PROMPT_PATH} \
    --exp_save_path         ${EXP_SAVE_PATH} \
    "

MODEL_ARGS="\
    --policy_ckpt_path          ${POLICY_CKPT_PATH} \
    --policy_tokenizer_path     ${POLICY_TOKENIZER_PATH} \
    --policy_model_path         ${POLICY_MODEL_PATH} \
    --rm_model_path             ${RM_MODEL_PATH} \
    "
    # --rm_model_path             ${RM_MODEL_PATH}/mp_rank_00_model_states.pt \
    # --rm_model_tokenizer_path   ${RM_TOKENIZER_PATH} \
    # --rm_config_path            ${RM_CONFIG_PATH}/config.json \

TRAINER_ARGS="\
    --do_validation     \
    --val_check_interval    5 \
    --val_size_per_task 20 \
    --policy_precision  bf16 \
    --learning_rate     3e-6 \
    --min_learning_rate 1e-7 \
    --total_steps       256 \
    --warmup_ratio      0.01 \
    --policy_train_batch_size   1 \
    --scheduler_type    cosine \
    "

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

mkdir ${LOG_ROOT_PATH}/exp/$MODEL_NAME/
mkdir ${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/
mkdir ${LOG_ROOT_PATH}/exp/$MODEL_NAME/pool/
mkdir ${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/pool/
mkdir ${LOG_ROOT_PATH}/exp/$MODEL_NAME/pool/$VERSION/
mkdir ${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/pool/$VERSION/


python3 -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE ${CODE_PATH}/${CODE_NAME} $options

set +x
