#!/bin/bash
#
#SBATCH --job-name=chatgpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=20 #
#SBATCH --gres=gpu:8                 # number of gpus
##SBATCH -x dgx047
##SBATCH --reservation=acagpt
#SBATCH -o outputs/%x-%j.log
#SBATCH -e outputs/%x-%j.err
# SBATCH --requeue
# SBATCH --qos=preemptive

GPUS_PER_NODE=8


MODEL_NAME=PPO_LLAMA_13B_STAGE4_1900_v4

CODE_PATH='/cognitive_comp/wanghao/experiments/chatgpt/chatgpt/examples'
CODE_NAME="train_llama_ppo_mp_sg_stage4.py"
CHATGPT_PATH="/cognitive_comp/wanghao/experiments/chatgpt/chatgpt/"

# Prompt数据集地址
# PROMPT_PATH="/cognitive_comp/songzhuoyang/processed_data/mixed_ppo_dataset_0423_for_llama/"
PROMPT_PATH="/cognitive_comp/wanghao/data/processed_data/mixed_ppo_dataset_0507_for_llama_13b"
# PROMPT_PATH="/cognitive_comp/wanghao/data/processed_data/mixed_ppo_dataset_0423_for_vicuna_llama/"
# 经验池保存地址
EXP_SAVE_PATH="/cognitive_comp/wanghao/experiments/workspace/chatgpt/llama_13b_rlhf/exp/"

# RL过程中Policy保存路径
POLICY_CKPT_PATH="/cognitive_comp/wanghao/experiments/workspace/chatgpt/llama_13b_rlhf/ckpt/$MODEL_NAME/"

# 初始Policy model的路径
POLICY_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/llama_stage4_step1900_MP_4"
# POLICY_TOKENIZER_PATH="/cognitive_comp/wanghao/models/llama_sft/llama_stage4_step1900_MP_4"
POLICY_TOKENIZER_PATH="/cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-c-pretrain-tokenizer/"
# REFERENCE_MODEL_PATH="/cognitive_comp/wanghao/models/llama_sft/13b_12000_FS"

RM_CONFIG_PATH="/cognitive_comp/sunqianguo/pretrained/checkpoints/7B/0405/v2/checkpoint-16000/"
RM_MODEL_PATH="/cognitive_comp/liangyuxin/workspace/rm_train/ckpt_log/RM_0426_Neg1layer/ckpt/last.ckpt/checkpoint"
RM_TOKENIZER_PATH="/cognitive_comp/songzhuoyang/models/llama_sft/20230405v1/"
# RM_CONFIG_PATH="/cognitive_comp/wanghao/models/reward_model"
# RM_MODEL_PATH="/cognitive_comp/wanghao/models/reward_model"
# RM_TOKENIZER_PATH="/cognitive_comp/wanghao/models/reward_model"

# sif镜像路径
# SIF_FILE="/cognitive_comp/songzhuoyang/images/fengshen_chatgpt.sif"


EXPERIMENT_ARGS="\
    --wandb_project     $MODEL_NAME \
    --seed              42 \
    "

PPO_ARGS="\
    --num_episodes      1000 \
    --max_timesteps     1 \
    --update_timesteps  1 \
    --sample_batch_size 64 \
    --buffer_limit_size 512 \
    --max_epoch_per_update  2 \
    --eps_clip          0.2 \
    --value_clip        0.2 \
    "

MEGATRON_ARGS="\
    --deepspeed_stage     1 \
    --tensor_model_parallel_size 4 \
    --pipe_model_parallel_size 1 \
    "
    # --offload_optimizer \

EXPERIENCE_ARGS="\
    --top_p                 0.85 \
    --top_k                 0 \
    --repetition_penalty    1. \
    --temperature           0.8 \
    --max_length            1024 \
    --kl_coef               0 \
    --experience_batch_size 64 \
    --policy_minibatch_size 16 \
    --rm_minibatch_size     8 \
    --rm_model_max_seq_len  1024 \
    --prompt_dataset_path   ${PROMPT_PATH} \
    --exp_save_path         ${EXP_SAVE_PATH} \
    "

MODEL_ARGS="\
    --policy_ckpt_path          ${POLICY_CKPT_PATH} \
    --policy_tokenizer_path     ${POLICY_TOKENIZER_PATH} \
    --policy_model_path         ${POLICY_MODEL_PATH} \
    --rm_config_path            ${RM_CONFIG_PATH}/config.json \
    --rm_model_path             ${RM_MODEL_PATH}/mp_rank_00_model_states.pt \
    --rm_model_tokenizer_path   ${RM_TOKENIZER_PATH} \
    "
    # --ref_model_path            ${REFERENCE_MODEL_PATH} \
    # --policy_ckpt_path          /opt/ckpt/ \
    # --policy_tokenizer_path     /opt/policy_tokenizer/ \
    # --policy_model_path         /opt/policy_model/ \
    # --rm_config_path            /opt/rm_config/config.json\
    # --rm_model_path             /opt/rm_model/mp_rank_00_model_states.pt \
    # --rm_model_tokenizer_path   /opt/rm_tokenizer/ \

TRAINER_ARGS="\
    --do_validation     \
    --val_check_interval    5 \
    --val_size_per_task 15 \
    --policy_precision  bf16 \
    --learning_rate     5e-6 \
    --min_learning_rate 1e-7 \
    --total_steps       512 \
    --warmup_ratio      0.01 \
    --policy_train_batch_size   1 \
    --scheduler_type    cosine \
    "

export options=" \
    $EXPERIMENT_ARGS \
    $MEGATRON_ARGS \
    $PPO_ARGS \
    $EXPERIENCE_ARGS \
    $MODEL_ARGS \
    $TRAINER_ARGS \
    "

export CMD="${CODE_PATH}/${CODE_NAME} $options"
echo "START"

python3 -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE ${CODE_PATH}/${CODE_NAME} $options
# singularity exec --nv \
#     -B /cognitive_comp:/cognitive_comp \
#     -B /home/wanghao:/home/wanghao \
#     $SIF_FILE python3 -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE ${CODE_PATH}/${CODE_NAME} $options
# #     # -B $CODE_PATH:/opt/code/ \
    # -B $CHATGPT_PATH:/opt/chatgpt/chatgpt/ \
    # -B $PROMPT_PATH:/opt/prompt_dataset/ \
    # -B $EXP_SAVE_PATH:/opt/exp/ \
    # -B $POLICY_CKPT_PATH:/opt/ckpt/ \
    # -B $POLICY_TOKENIZER_PATH:/opt/policy_tokenizer/ \
    # -B $POLICY_MODEL_PATH:/opt/policy_model/ \
    # -B $RM_CONFIG_PATH:/opt/rm_config/ \
    # -B $RM_MODEL_PATH:/opt/rm_model/ \
    # -B $RM_TOKENIZER_PATH:/opt/rm_tokenizer/ \
    # $SIF_FILE python3 -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE /opt/code/$CODE_NAME $options

set +x
