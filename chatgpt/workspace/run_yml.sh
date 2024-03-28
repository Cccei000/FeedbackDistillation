#!/bin/bash
#
#SBATCH --job-name=pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64 
#SBATCH --mem-per-cpu=8G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:8                # number of gpus
#SBATCH -p pog
#SBATCH -o ./%x-%j.log
#SBATCH -e ./%x-%j.err

GPUS_PER_NODE=8
MASTER_PORT=45100

CODE_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt/"
CODE_NAME="launch_pipeline.py"

##### 指定配置文件 #####
YAML_PATH="/cognitive_comp/liangyuxin/chatgpt/chatgpt/pipeline/scripts/token_level_full_lite.yml"
#######################

CONFIG_ATGS="\
    --config $YAML_PATH \
    "

export options=" \
    $CONFIG_ATGS \
    "

export CMD="${CODE_PATH}/${CODE_NAME} $options"
echo "START"

python3 -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE --master_port $MASTER_PORT ${CODE_PATH}/${CODE_NAME} $options

set +x
