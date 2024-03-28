#!/bin/sh
#SBATCH --job-name=CONVERT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:hgx:1 -p pog-preempted
#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G is default)
#SBATCH -o ./%x-%j.log
#SBATCH -e ./%x-%j.err

INPUT_PATH=/cognitive_comp/liangyuxin/workspace/pipeline/RM_LLAMA2_13B_0829/0829/ckpt/reward_model/global_step14005_hf/
OUTPUT_PATH=/cognitive_comp/liangyuxin/workspace/pipeline/RM_LLAMA2_13B_0829/0829/ckpt/reward_model/global_step14005_fs_mp4_test/
CODE_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt/tools"
# sif镜像路径
SIF_FILE="/cognitive_comp/liangyuxin/images/fengshen_chatgpt_0920.sif"
CHATGPT_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt"
FENSHEN_INNER="/cognitive_comp/liangyuxin/fengshen_inner/fengshenbang-lm/fengshen_inner"

ARGS="\
    --input_path /opt/input_path \
    --output_path /opt/output_path/ \
    --model_parallel_size 4 \
    --is_rm \
    --from_hf \
    "
    # --is_multi_head_rm \

mkdir -p $OUTPUT_PATH

export options=" \
    $ARGS \
    "
echo "START"

singularity exec --nv \
    -B $INPUT_PATH:/opt/input_path \
    -B $OUTPUT_PATH:/opt/output_path/ \
    -B $CODE_PATH:/opt/code/ \
    -B $CHATGPT_PATH:/opt/chatgpt/chatgpt/ \
    -B $FENSHEN_INNER:/opt/fengshenbang-lm/fengshen_inner/ \
    $SIF_FILE python3 /opt/code/convert_hf_llama_to_fs_mp.py $options