#!/bin/sh
#SBATCH --job-name=RM_LORA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=100
#SBATCH --gres=gpu:hgx:8 -p pog # -preempted
#SBATCH --mem-per-cpu=8G # memory per cpu-core (4G is default)
#SBATCH -o ./%x-%j.log
#SBATCH -e ./%x-%j.err

GPUS_PER_NODE=8

MODEL_NAME=RM_LLAMA_13B_0629_lora
# 训练入口：
CODE_PATH='/cognitive_comp/liangyuxin/chatgpt/chatgpt/examples'
CODE_NAME="train_llama_rm_lora.py"
CHATGPT_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt"

# 数据集地址
DATASET_PATH="/cognitive_comp/liangyuxin/datasets/RM/0621/merged_dataset"

# RM保存路径
RM_CKPT_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/$MODEL_NAME/"
# 初始RM&tokenizer路径
RM_TOKENIZER_PATH="/cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-c-pretrain-tokenizer/"
RM_MODEL_PATH="/cognitive_comp/liangyuxin/models/SFT/13B/merged_average-chat_19000_18000-math_6000_fs_MP4"

# sif镜像&fenshen框架路径
SIF_FILE="/cognitive_comp/liangyuxin/images/fengshen_chatgpt.sif"
FENSHEN_INNER="/cognitive_comp/liangyuxin/fengshen_inner/fengshenbang-lm/fengshen_inner"

EXPERIMENT_ARGS="\
    --wandb_project     $MODEL_NAME \
    --seed              42 \
    "

MEGATRON_ARGS="\
    --deepspeed_stage     2 \
    --tensor_model_parallel_size 4 \
    --pipe_model_parallel_size 1 \
    --gradient_accumulation_steps 8 \
    "
    # --offload_optimizer \

DATA_ARGS="\
    --dataset_path /opt/dataset/  \
    "

MODEL_ARGS="\
    --rm_model_path             /opt/rm_model/ \
    --rm_tokenizer_path         /opt/rm_tokenizer \
    --rm_ckpt_path              /opt/ckpt/ \
    --max_length                2048 \
    "

TRAINER_ARGS="\
    --rm_batch_size      1 \
    --val_check_interval 0.01 \
    --learning_rate      1e-5 \
    --min_learning_rate  1e-7 \
    --warmup_ratio       0.01 \
    --weight_decay       0.01 \
    --policy_precision   bf16 \
    --max_epochs         2 \
    "
    # --activation_checkpointing \
    
export options=" \
    $EXPERIMENT_ARGS \
    $MEGATRON_ARGS \
    $DATA_ARGS \
    $MODEL_ARGS \
    $TRAINER_ARGS \
    "
export OMP_NUM_THREADS=12

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
    $SIF_FILE python3 -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE --master_port 25641 /opt/code/$CODE_NAME $options