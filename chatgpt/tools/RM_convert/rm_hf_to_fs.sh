#!/bin/sh
#SBATCH --job-name=CONVERT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:hgx:1 -p pog
#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G is default)
#SBATCH -o ./%x-%j.log
#SBATCH -e ./%x-%j.err

GPUS_PER_NODE=1

PRETRAINED_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0801_multi/global_step14424_hf"
OUTPUT_PATH='/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0801_multi/global_step14424_fs'
CODE_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt/utils/llama_convert/RM_convert"
# sif镜像路径
SIF_FILE="/cognitive_comp/liangyuxin/images/fengshen_chatgpt.sif"
CHATGPT_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt"
FENSHEN_INNER="/cognitive_comp/liangyuxin/fengshen_inner/fengshenbang-lm/fengshen_inner"

mkdir $OUTPUT_PATH

ARGS="\
    --pretrained_path      /opt/pretrained_model_path\
    --tokenizer_path       /opt/pretrained_model_path\
    --output_path          /opt/output_path/ \
    --multiplier           8 \
    "

export options=" \
    $ARGS \
    "
echo "START"


singularity exec --nv \
    -B $PRETRAINED_PATH:/opt/pretrained_model_path \
    -B $OUTPUT_PATH:/opt/output_path/ \
    -B $CODE_PATH:/opt/code/ \
    -B $CHATGPT_PATH:/opt/chatgpt/chatgpt/ \
    -B $FENSHEN_INNER:/opt/fengshenbang-lm/fengshen_inner/ \
    $SIF_FILE python3 /opt/code/rm_hf_to_fs.py $options
