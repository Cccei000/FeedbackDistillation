#!/bin/sh
#SBATCH --job-name=RM_TEST
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=100
#SBATCH --gres=gpu:hgx:8 -p pog-preempted
#SBATCH --mem-per-cpu=8G # memory per cpu-core (4G is default)
#SBATCH -o ./%x-%j.log
#SBATCH -e ./%x-%j.err

GPUS_PER_NODE=8

MODEL_NAME=RM_LLAMA_65B_0621

# 数据集地址
DATASET_PATH="/cognitive_comp/liangyuxin/datasets/RM/0621/merged_dataset"
# RM保存路径
RM_CKPT_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/65B_RM/ckpt/$MODEL_NAME/"
# 初始RM路径
RM_TOKENIZER_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/llama_65b_hf_model_sft_79w_v2_fs"
RM_MODEL_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/65B_RM/ckpt/RM_LLAMA_65B_0625/global_step140531_fs_MP8"


# sif镜像路径
SIF_FILE="/cognitive_comp/liangyuxin/images/fengshen_chatgpt.sif"
# 训练入口：
CODE_PATH='/cognitive_comp/liangyuxin/chatgpt/chatgpt/examples'
CODE_NAME="test_rm_mp.py"
CHATGPT_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt"
FENSHEN_INNER="/cognitive_comp/liangyuxin/fengshen_inner/fengshenbang-lm/fengshen_inner"

EXPERIMENT_ARGS="\
    --wandb_project     $MODEL_NAME \
    --seed              42 \
    "

MEGATRON_ARGS="\
    --deepspeed_stage     2 \
    --tensor_model_parallel_size 8 \
    --pipe_model_parallel_size 1 \
    --gradient_accumulation_steps 8 \
    "

DATA_ARGS="\
    --dataset_path /opt/dataset/  \
    --prefix_user Human:  \
    --prefix_bot \nAssistant:  \
    "

MODEL_ARGS="\
    --rm_model_path             /opt/rm_model/ \
    --rm_tokenizer_path         /opt/rm_tokenizer \
    --rm_ckpt_path              /opt/ckpt/ \
    --max_length                2048 \
    --model_type                llama_65b \
    "
TRAINER_ARGS="\
    --rm_batch_size      1 \
    --val_check_interval 0.05 \
    --learning_rate      5e-5 \
    --min_learning_rate  1e-7 \
    --warmup_ratio       0.01 \
    --policy_precision   bf16 \
    --weight_decay       0.01 \
    --max_epochs         2 \
    --load_from_rm \
    "
    # --activation_checkpointing \

export options=" \
    $EXPERIMENT_ARGS \
    $MEGATRON_ARGS \
    $DATA_ARGS \
    $MODEL_ARGS \
    $TRAINER_ARGS \
    "

echo "START"

mkdir $RM_CKPT_PATH

singularity exec --nv \
    -B $CODE_PATH:/opt/code/ \
    -B $CHATGPT_PATH:/opt/chatgpt/chatgpt/ \
    -B $DATASET_PATH:/opt/dataset/ \
    -B $RM_CKPT_PATH:/opt/ckpt/ \
    -B $RM_TOKENIZER_PATH:/opt/rm_tokenizer/ \
    -B $RM_MODEL_PATH:/opt/rm_model/ \
    -B $FENSHEN_INNER:/opt/fengshenbang-lm/fengshen_inner/ \
    $SIF_FILE python3 -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE /opt/code/$CODE_NAME $options
    