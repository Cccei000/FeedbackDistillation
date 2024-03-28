#!/bin/sh
#SBATCH --job-name=CONVERT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:hgx:1 -p pog-preempted
#SBATCH --mem-per-cpu=16G # memory per cpu-core (4G is default)
#SBATCH -o ./log/%x-%j.log
#SBATCH -e ./log/%x-%j.err

GPUS_PER_NODE=1

# PRETRAINED_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/llama_65b_hf_model_sft_79w_v2_fs"
# CKPT_PATH='/cognitive_comp/liangyuxin/workspace/chatgpt/65B_RM/ckpt/RM_LLAMA_65B_0602/global_step79520'
# OUTPUT_PATH='/cognitive_comp/liangyuxin/workspace/chatgpt/65B_RM/ckpt/RM_LLAMA_65B_0602/global_step79520_hf'
PRETRAINED_PATH="/cognitive_comp/wanghao/models/llama_sft/v2_stage2.2_step6600_fs/"
CKPT_PATH='/cognitive_comp/zejianxie/idea-best'
OUTPUT_PATH='/cognitive_comp/shenjunyi/g/model/math/ziya-llama_math_new_hf'
CONFIG_PATH='/cognitive_comp/wanghao/models/llama_sft/v2_stage2.2_step6600_fs/config.json'

CODE_PATH="/cognitive_comp/shenjunyi/g/chatgpt/chatgpt/tools"
python3 ${CODE_PATH}/convert_fs_mp_to_hf_llama.py -i ${CKPT_PATH} -o ${OUTPUT_PATH} -c ${CONFIG_PATH}
