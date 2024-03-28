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

# INPUT_DIR="/cognitive_comp/liangyuxin/workspace/chatgpt/65B_RM/ckpt/RM_LLAMA_65B_0625/global_step144134_fs"
# OUTPUT_PATH='/cognitive_comp/liangyuxin/workspace/chatgpt/65B_RM/ckpt/RM_LLAMA_65B_0625/global_step144134_fs_MP8'
INPUT_DIR="/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0801_multi/global_step14424_fs"
OUTPUT_PATH='/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0801_multi/global_step14424_fs_MP4'
CODE_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt/utils/llama_convert/RM_convert"

# sif镜像路径
SIF_FILE="/cognitive_comp/liangyuxin/images/fengshen_chatgpt.sif"
CHATGPT_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt"
FENSHEN_INNER="/cognitive_comp/liangyuxin/fengshen_inner/fengshenbang-lm/fengshen_inner"

mkdir $OUTPUT_PATH

ARGS="\
    --input_dir             /opt/input_dir \
    --output_dir            /opt/output_path/ \
    --model_parallel_size   4 \
    "

export options=" \
    $ARGS \
    "
echo "START"


singularity exec --nv \
    -B $INPUT_DIR:/opt/input_dir \
    -B $OUTPUT_PATH:/opt/output_path/ \
    -B $CODE_PATH:/opt/code/ \
    -B $CHATGPT_PATH:/opt/chatgpt/chatgpt/ \
    -B $FENSHEN_INNER:/opt/fengshenbang-lm/fengshen_inner/ \
    $SIF_FILE python3 /opt/code/rm_fs_to_llama_tp.py $options
