#!/bin/bash
#SBATCH --job-name=RM_TEST
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64 
#SBATCH --mem-per-cpu=8G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:1               # number of gpus
#SBATCH -p pog-preempted
#SBATCH -o ./%x-%j.log
#SBATCH -e ./%x-%j.err

python3 /cognitive_comp/liangyuxin/chatgpt/chatgpt/test/test_hf_rm.py
