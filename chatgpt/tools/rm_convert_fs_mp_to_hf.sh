#!/bin/sh
#SBATCH --job-name=CONVERT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:hgx:1 -p pog-preempted
#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G is default)
#SBATCH -o ./%x-%j.log
#SBATCH -e ./%x-%j.err

INPUT_PATH=/cognitive_comp/liangyuxin/workspace/pipeline/RM_LLAMA2_13B_0914/ckpt/reward_model/global_step10941
OUTPUT_PATH=/cognitive_comp/liangyuxin/workspace/pipeline/RM_LLAMA2_13B_0914/ckpt/reward_model/global_step10941_hf/
CONFIG_PATH=/cognitive_comp/liangyuxin/workspace/pipeline/RM_LLAMA2_13B_0914/models/policy
CODE_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt/tools"
# sif镜像路径
SIF_FILE="/cognitive_comp/liangyuxin/images/fengshen_chatgpt.sif"
CHATGPT_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt"

ARGS="\
    --input_path /opt/input_path \
    --output_path /opt/output_path/ \
    --fs_config_path /opt/config_path/config.json \
    --model_parallel_size 4 \
    --is_rm \
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
    -B $CONFIG_PATH:/opt/config_path/ \
    -B $CODE_PATH:/opt/code/ \
    -B $CHATGPT_PATH:/opt/chatgpt/chatgpt/ \
    $SIF_FILE python3 /opt/code/convert_fs_mp_to_hf_llama.py $options