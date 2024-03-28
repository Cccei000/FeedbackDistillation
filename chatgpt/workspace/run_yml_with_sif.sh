#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64 
#SBATCH --mem-per-cpu=8G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:8       # number of gpus
#SBATCH -p pog-preempted
#SBATCH -o ./%x-%j.log
#SBATCH -e ./%x-%j.err

GPUS_PER_NODE=8
# MASTER_PORT=45100

CODE_PATH="/cognitive_comp/liangyuxin/chatgpt/"
CODE_NAME="/chatgpt/launch_pipeline.py"
# sif镜像&fenshen框架路径
SIF_FILE="/cognitive_comp/liangyuxin/images/fengshen_chatgpt_0920.sif"
FENSHEN_INNER="/cognitive_comp/liangyuxin/fengshen_inner/fengshenbang-lm/fengshen_inner"
CHATGPT_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt"

DATASET_PATH="/cognitive_comp/liangyuxin/datasets/RM/0918/"
WORKSPACE_PATH="/cognitive_comp/liangyuxin/workspace/pipeline/RM_LLAMA2_13B_Pipeline_debug"
TOKENINZER_PATH="/cognitive_comp/yangping/checkpoints/llama2/neox2hf/llama2_hf_13b_step136000"
POLICY_PATH="/cognitive_comp/yangping/checkpoints/llama2/neox2hf/llama2_hf_13b_step136000"

mkdir -p $WORKSPACE_PATH
##### 指定配置文件 #####
# YAML_PATH="/opt/code/test/rm_llama_sample_sif.yml"
# YAML_PATH="/opt/code/test/rm_llama_token_sif.yml"
YAML_PATH="/opt/code/test/rm_llama_sample_token_mix_sif.yml"


#######################

CONFIG_ATGS="\
    --config $YAML_PATH \
    "

export options=" \
    $CONFIG_ATGS \
    "

export CMD="${CODE_PATH}/${CODE_NAME} $options"
echo "START"

singularity exec --nv \
    -B $CODE_PATH:/opt/code/ \
    -B $WORKSPACE_PATH:/opt/workspace/ \
    -B $DATASET_PATH:/opt/dataset/ \
    -B $TOKENINZER_PATH:/opt/tokenizer/ \
    -B $POLICY_PATH:/opt/policy/ \
    -B $CHATGPT_PATH:/opt/chatgpt/chatgpt/ \
    -B $FENSHEN_INNER:/opt/fengshenbang-lm/fengshen_inner/ \
    $SIF_FILE python3 -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE --master_port 25641 /opt/code/$CODE_NAME $options

set +x
