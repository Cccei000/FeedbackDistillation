#!/bin/bash
#
#SBATCH --job-name=chatgpt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=180 #
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH -x dgx047
#SBATCH --reservation=acagpt
#SBATCH -o outputs/%x-%j.log
#SBATCH -e outputs/%x-%j.err

NNODES=2
GPUS_PER_NODE=8

MASTER_ADDR=`perl -le '$_=$ENV{"SLURM_JOB_NODELIST"}; s/,.*//; s/-.*//; s/\[//; print' | xargs -I {} host $n {} | awk '{print $4}'`
MASTER_PORT=15935
echo 'master_addr '$MASTER_ADDR
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

MODEL_NAME=PPO_LLAMA13B_S4_1900_RM13B0510
VERSION="0512_1447"

CODE_PATH='/cognitive_comp/wanghao/experiments/chatgpt/chatgpt/examples'
CODE_NAME="train_llama_ppo_13bSFT_13bRM.py"

LOG_ROOT_PATH='/cognitive_comp/wanghao/experiments/workspace/chatgpt/llama_13b_rlhf'
# Prompt数据集地址
PROMPT_PATH="/cognitive_comp/wanghao/data/processed_data/mixed_ppo_dataset_0423_for_llama_13b/"
# 经验池保存地址
EXP_SAVE_PATH="${LOG_ROOT_PATH}/exp/$MODEL_NAME/pool/$VERSION/"

# RL过程中Policy保存路径
POLICY_CKPT_PATH="${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/pool/$VERSION/"

# 初始Policy model的路径
POLICY_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/llama_stage4_step1900_MP_8"
POLICY_TOKENIZER_PATH="/cognitive_comp/wanghao/models/llama_sft/13B-c-pretrain-tokenizer"

# RM_CONFIG_PATH="/cognitive_comp/sunqianguo/pretrained/checkpoints/7B/0405/v2/checkpoint-16000/"
RM_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/rm_13b_0510_MP8"
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
    --sample_batch_size 32 \
    --buffer_limit_size 512 \
    --max_epoch_per_update  2 \
    --eps_clip          0.2 \
    --value_clip        0.2 \
    --gamma             1.0 \
    --lam               0.95 \
    "
    # --enable_pool        \

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
    --temperature           1. \
    --max_length            1024 \
    --kl_coef               0 \
    --experience_batch_size 32 \
    --policy_minibatch_size 2 \
    --rm_minibatch_size     8 \
    --rm_model_max_seq_len  1024 \
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
    --learning_rate     1e-5 \
    --min_learning_rate 1e-7 \
    --total_steps       256 \
    --warmup_ratio      0.05 \
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


mkdir ${LOG_ROOT_PATH}/exp/$MODEL_NAME/
mkdir ${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/
mkdir ${LOG_ROOT_PATH}/exp/$MODEL_NAME/pool/
mkdir ${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/pool/
mkdir ${LOG_ROOT_PATH}/exp/$MODEL_NAME/pool/$VERSION/
mkdir ${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/pool/$VERSION/

export LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --max_restarts 0\
    "

export CMD="${CODE_PATH}/${CODE_NAME} $options"
echo "START"

srun -N $NNODES --gres=gpu:$GPUS_PER_NODE --ntasks-per-node=1 --cpus-per-task=180 -o outputs/%x-%j.log -e outputs/%x-%j.err bash -c '$LAUNCHER  --node_rank ${SLURM_PROCID} $CMD'

set +x
