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

CKPT_PATH='/cognitive_comp/songzhuoyang/models/Llama-2-13B-fp16'
OUTPUT_PATH='/cognitive_comp/liangyuxin/workspace/pipeline/RM_LLAMA2_EN_13B_0906/models/policy'

CODE_PATH="/cognitive_comp/shenjunyi/g/chatgpt/chatgpt/tools"
python3 ${CODE_PATH}/convert_hf_llama_to_fs_mp.py -i ${CKPT_PATH} -o ${OUTPUT_PATH}  -mp 4 --from_hf
