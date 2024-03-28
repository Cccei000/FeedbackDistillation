#!/bin/sh
#SBATCH --job-name=RM_DATA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:hgx:1 
#SBATCH -p pog-preempted
#SBATCH --mem-per-cpu=32G # memory per cpu-core (4G is default)
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.err

SAVE_PATH="/cognitive_comp/liangyuxin/datasets/RM/0918/"
mkdir -p $SAVE_PATH
ARGS="\
        --save_base_path $SAVE_PATH \
        --data_config /cognitive_comp/liangyuxin/chatgpt/chatgpt/tools/RM_dataset_construct/dataset_config.json \
        --backup_ds_path /cognitive_comp/liangyuxin/datasets/RM/0918_backup/ \
        "
        # --debias \
        
export options=" \
        $ARGS \
        "

export SCRIPT_PATH=/cognitive_comp/liangyuxin/chatgpt/chatgpt/tools/RM_dataset_construct/construct_rm_dataset.py


python3 $SCRIPT_PATH $options