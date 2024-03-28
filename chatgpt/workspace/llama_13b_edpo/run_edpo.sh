#!/bin/bash
#
#SBATCH --job-name=chatgpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=12 #
#SBATCH --gres=gpu:hgx:4                 # number of gpus
#SBATCH -w hgx021
# SBATCH -p pog-preempted
#SBATCH -p pog
##SBATCH --reservation=acagpt
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.err
# SBATCH --requeue
# SBATCH --qos=preemptive

GPUS_PER_NODE=4

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=13237

MODEL_NAME=EDPO_LLAMA13B_0727Preference
VERSION="0808_1400_poem_small"

CODE_PATH='/cognitive_comp/wanghao/experiments/chatgpt/chatgpt/examples'
CODE_NAME="train_llama_edpo_13bSFT_13bRM.py"

LOG_ROOT_PATH='/cognitive_comp/wanghao/experiments/workspace/chatgpt/llama_13b_edpo'
# Prompt数据集地址


# PROMPT_PATH="/cognitive_comp/wanghao/data/processed_data/edpo_dataset_0727_for_llama_13b"
PROMPT_PATH="/cognitive_comp/wanghao/data/processed_data/edpo_dataset_poem_10_0808"
# 经验池保存地址
EXP_SAVE_PATH="${LOG_ROOT_PATH}/exp/$MODEL_NAME/$VERSION/"

# RL过程中Policy保存路径
POLICY_CKPT_PATH="${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/$VERSION/"

# 初始Policy model的路径
POLICY_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/merged_average-chat_19000-mmm_0615_ind_chat_19000_math_6000-mer_0619_ind_chat_19000_18000_math_6000_fs_MP4"
# POLICY_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/llama_13B_v2_s2.2_s3_s3.1_rlhf_0601_RM13B0525_step44_MP4"
# POLICY_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/Ziya-Writing-13B-0706_fs_MP4"
# POLICY_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/Ziya-Writing-13B-v1_fs_MP4"
POLICY_TOKENIZER_PATH="/cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-c-pretrain-tokenizer/"

# RM_CONFIG_PATH="/cognitive_comp/sunqianguo/pretrained/checkpoints/7B/0405/v2/checkpoint-16000/"
# RM_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/rm_13b_0510_MP4"
# RM_MODEL_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0520/global_step55666_fs"
# RM_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/rm_13b_0525_MP4"
# RM_MODEL_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0621_para/global_step72074_fs_MP4"
RM_MODEL_PATH="/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0707_ALL/global_step33581_fs_MP4"

# sif镜像路径
# SIF_FILE="/cognitive_comp/songzhuoyang/images/fengshen_chatgpt.sif"


EXPERIMENT_ARGS="\
    --wandb_project     $MODEL_NAME \
    --wandb_team        yukawa \
    --wandb_name        "$VERSION" \
    --seed              42 \
    "

EDPO_ARGS="\
    --num_episodes      2048 \
    --max_timesteps     1 \
    --update_timesteps  1 \
    --sample_batch_size 1 \
    --buffer_limit_size 512 \
    --max_epoch_per_update  1 \
    --edpo_preference_batch_size 1 \
    --dpo_beta          0.5 \
    --save_every_n_episode 50 \
    "
    # --ignore_ref_first_n_steps   10 \
    # --enable_pool        \

MEGATRON_ARGS="\
    --deepspeed_stage     1 \
    --tensor_model_parallel_size 4 \
    --pipe_model_parallel_size 1 \
    --offload_optimizer \
    "
    # --activation_checkpointing \

    # --max_new_tokens        512 \
EXPERIENCE_ARGS="\
    --top_p                 0.85 \
    --top_k                 0 \
    --repetition_penalty    1. \
    --temperature           1.0 \
    --max_length            2048 \
    --experience_batch_size 1 \
    --gen_minibatch_size    1 \
    --policy_minibatch_size 1 \
    --rm_minibatch_size     2 \
    --rm_model_max_seq_len  2048 \
    --prompt_dataset_path   ${PROMPT_PATH} \
    --exp_save_path         ${EXP_SAVE_PATH} \
    "
    # --enabling_tot  \
    # --gs_gen_batch_size     64 \
    # --gs_eval_batch_size    16 \
    # --gs_gen_repeat_times   3 \
    # --gs_breadth            2 \
    # --gs_iterations         2 \

MODEL_ARGS="\
    --policy_ckpt_path          ${POLICY_CKPT_PATH} \
    --policy_tokenizer_path     ${POLICY_TOKENIZER_PATH} \
    --policy_model_path         ${POLICY_MODEL_PATH} \
    --rm_model_path             ${RM_MODEL_PATH} \
    "

TRAINER_ARGS="\
    --do_validation     \
    --val_check_interval    5 \
    --val_size_per_task 10 \
    --policy_precision  bf16 \
    --learning_rate     2e-6 \
    --min_learning_rate 1e-7 \
    --total_steps       512 \
    --warmup_ratio      0.01 \
    --policy_train_batch_size   1 \
    --scheduler_type    cosine \
    --clip_grad         \
    "

export options=" \
    $EXPERIMENT_ARGS \
    $MEGATRON_ARGS \
    $EDPO_ARGS \
    $EXPERIENCE_ARGS \
    $MODEL_ARGS \
    $TRAINER_ARGS \
    "

export CMD="${CODE_PATH}/${CODE_NAME} $options"
echo "START"

mkdir ${LOG_ROOT_PATH} 
mkdir ${LOG_ROOT_PATH}/exp 
mkdir ${LOG_ROOT_PATH}/ckpt 
mkdir ${LOG_ROOT_PATH}/exp/$MODEL_NAME/
mkdir ${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/
mkdir ${LOG_ROOT_PATH}/exp/$MODEL_NAME/$VERSION/
mkdir ${LOG_ROOT_PATH}/ckpt/$MODEL_NAME/$VERSION/


srun python3 -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE --master_port $MASTER_PORT ${CODE_PATH}/${CODE_NAME} $options

set +x
