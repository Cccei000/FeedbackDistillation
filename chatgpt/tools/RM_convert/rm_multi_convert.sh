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

# 13B
PRETRAINED_PATH="//cognitive_comp/wanghao/models/llama_sft/v2_stage2.2_step6600_fs/"
CKPT_PATH='/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0801_multi/global_step27458'
HF_PATH='/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0801_multi/global_step27458_hf'
FS_PATH='/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0801_multi/global_step27458_fs'
FS_MP_PATH='/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0801_multi/global_step27458_fs_MP4'

mkdir $HF_PATH
mkdir $FS_PATH
mkdir $FS_MP_PATH

CODE_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt/utils/llama_convert/RM_convert"
# sif镜像路径
SIF_FILE="/cognitive_comp/liangyuxin/images/fengshen_chatgpt.sif"
CHATGPT_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt"
FENSHEN_INNER="/cognitive_comp/liangyuxin/fengshen_inner/fengshenbang-lm/fengshen_inner"


echo "CONVERT TO HF"
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
singularity exec --nv \
    -B $PRETRAINED_PATH:/opt/pretrained_model_path \
    -B $CKPT_PATH:/opt/ckpt_path/ \
    -B $HF_PATH:/opt/output_path/ \
    -B $CODE_PATH:/opt/code/ \
    -B $CHATGPT_PATH:/opt/chatgpt/chatgpt/ \
    $SIF_FILE python3 /opt/code/rm_fs_to_hf.py $options


echo "CONVERT TO FS"
ARGS="\
    --pretrained_path      /opt/pretrained_model_path\
    --tokenizer_path       /opt/pretrained_model_path\
    --output_path          /opt/output_path/ \
    --multiplier           8 \
    --multi_value_head         \
    "

export options=" \
    $ARGS \
    "
singularity exec --nv \
    -B $HF_PATH:/opt/pretrained_model_path \
    -B $FS_PATH:/opt/output_path/ \
    -B $CODE_PATH:/opt/code/ \
    -B $CHATGPT_PATH:/opt/chatgpt/chatgpt/ \
    -B $FENSHEN_INNER:/opt/fengshenbang-lm/fengshen_inner/ \
    $SIF_FILE python3 /opt/code/rm_hf_to_fs.py $options


echo "CONVERT TO FS_MP"

ARGS="\
    --input_dir             /opt/input_dir \
    --output_dir            /opt/output_path/ \
    --model_parallel_size   4 \
    "

export options=" \
    $ARGS \
    "
singularity exec --nv \
    -B $FS_PATH:/opt/input_dir \
    -B $FS_MP_PATH:/opt/output_path/ \
    -B $CODE_PATH:/opt/code/ \
    -B $CHATGPT_PATH:/opt/chatgpt/chatgpt/ \
    -B $FENSHEN_INNER:/opt/fengshenbang-lm/fengshen_inner/ \
    $SIF_FILE python3 /opt/code/rm_fs_to_llama_tp.py $options