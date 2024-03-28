#!/bin/sh
#SBATCH --job-name=CONVERT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:hgx:1 -p pog-preempted
#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G is default)
#SBATCH -o ./%x-%j.log
#SBATCH -e ./%x-%j.err

GPUS_PER_NODE=1

# PRETRAINED_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/llama_65b_hf_model_sft_79w_v2_fs"
# CKPT_PATH='/cognitive_comp/liangyuxin/workspace/chatgpt/65B_RM/ckpt/RM_LLAMA_65B_0602/global_step79520'
# OUTPUT_PATH='/cognitive_comp/liangyuxin/workspace/chatgpt/65B_RM/ckpt/RM_LLAMA_65B_0602/global_step79520_hf'
PRETRAINED_PATH="//cognitive_comp/wanghao/models/llama_sft/v2_stage2.2_step6600_fs/"
CKPT_PATH='/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0801_multi/global_step14424'
OUTPUT_PATH='/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0801_multi/global_step14424_hf'

CODE_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt/utils/llama_convert/RM_convert"
# sif镜像路径
SIF_FILE="/cognitive_comp/liangyuxin/images/fengshen_chatgpt.sif"
CHATGPT_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt"
mkdir $OUTPUT_PATH

ARGS="\
    --pretrained_model_path    /opt/pretrained_model_path\
    --ckpt_path                /opt/ckpt_path/ \
    --output_path              /opt/output_path/ \
    --model_parallel_size      4 \
    --multi_value_head         \
    "
    # --is_lora \

export options=" \
    $ARGS \
    "
echo "START"


singularity exec --nv \
    -B $PRETRAINED_PATH:/opt/pretrained_model_path \
    -B $CKPT_PATH:/opt/ckpt_path/ \
    -B $OUTPUT_PATH:/opt/output_path/ \
    -B $CODE_PATH:/opt/code/ \
    -B $CHATGPT_PATH:/opt/chatgpt/chatgpt/ \
    $SIF_FILE python3 /opt/code/rm_fs_to_hf.py $options
