#!/bin/bash
#
#SBATCH --job-name=infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8 #
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH -p pog
#SBATCH -o outputs/%x-%j.log
#SBATCH -e outputs/%x-%j.err


GPUS_PER_NODE=4

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=45530

MODEL_NAME=PPOPP_Writing_LLAMA13BSFT_S2_S2.2_S3.1_RM13B0707
VERSION="test"

CODE_PATH='/cognitive_comp/songzhuoyang/projects/chatgpt/chatgpt/examples'
CODE_NAME="inference_llama_ppo_13bRLHF.py"

LOG_ROOT_PATH='/cognitive_comp/songzhuoyang/workspace/chatgpt/13B_rlhf' # Prompt数据集地址

PROMPT_PATH="/cognitive_comp/songzhuoyang/processed_data/evaluation_writing/"

# 经验池保存地址
EXP_SAVE_PATH="/cognitive_comp/songzhuoyang/processed_data/ppopp_0721_writing_test"

# RL过程中Policy保存路径
POLICY_CKPT_PATH="/cognitive_comp/songzhuoyang/workspace/chatgpt/13B_rlhf/ckpt/PPOPP_Writing_LLAMA13BSFT_S2_S2.2_S3.1_RM13B0707/pool/0719_1002_sft0719/global_step99"

# 初始Policy model的路径
POLICY_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/merged_average-chat_19000-mmm_0615_ind_chat_19000_math_6000-mer_0619_ind_chat_19000_18000_math_6000_fs_MP4"
POLICY_TOKENIZER_PATH="/cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-c-pretrain-tokenizer/"
RM_MODEL_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0707_ALL/global_step33581_fs_MP4"

# sif镜像路径
# SIF_FILE="/cognitive_comp/songzhuoyang/images/fengshen_chatgpt.sif"

MEGATRON_ARGS="\
    --adam_epsilon        1e-5 \
    --deepspeed_stage     1 \
    --tensor_model_parallel_size 4 \
    --pipe_model_parallel_size 1 \
    --offload_optimizer \
    "
    # 

EXPERIENCE_ARGS="\
    --top_p                 0.85 \
    --top_k                 0 \
    --repetition_penalty    1. \
    --temperature           1.0 \
    --max_new_tokens        2048 \
    --max_length            2048 \
    --gen_minibatch_size    64 \
    --rm_minibatch_size     2 \
    --rm_model_max_seq_len  2048 \
    --prompt_dataset_path   ${PROMPT_PATH} \
    --exp_save_path         ${EXP_SAVE_PATH} \
    "
    
    # --gen_minibatch_size    16 \
    # --policy_minibatch_size 8 \
    # --rm_minibatch_size     2 \

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
    --seed              42 \
    --do_validation     \
    --val_check_interval    5 \
    --val_size_per_task 128 \
    --policy_precision  bf16 \
    --learning_rate     2e-6 \
    --min_learning_rate 1e-7 \
    --total_steps       1024 \
    --warmup_ratio      0.01 \
    --policy_train_batch_size   4 \
    --scheduler_type    cosine \
    --entropy_loss_coef 0.01 \
    --entropy_loss_decay_rate 1.0 \
    --clip_grad         \
    "

export options=" \
    $MEGATRON_ARGS \
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


python3 -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE --master_port $MASTER_PORT ${CODE_PATH}/${CODE_NAME} $options

set +x
