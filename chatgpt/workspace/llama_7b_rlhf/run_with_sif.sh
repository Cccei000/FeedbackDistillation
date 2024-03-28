
GPUS_PER_NODE=8

MODEL_NAME=PPO_LLAMA_DEBUG

# 训练入口： /cognitive_comp/songzhuoyang/projects/chatgpt/chatgpt/examples/train_chatglm_ppo.py
CODE_PATH='/cognitive_comp/songzhuoyang/projects/chatgpt/chatgpt/examples'
CODE_NAME="train_llama_ppo.py"
CHATGPT_PATH="/cognitive_comp/songzhuoyang/projects/chatgpt/chatgpt/"

# Prompt数据集地址
PROMPT_PATH="/cognitive_comp/songzhuoyang/processed_data/mixed_ppo_dataset_0327_for_llama/"
# 经验池保存地址
EXP_SAVE_PATH="/cognitive_comp/songzhuoyang/workspace/chatgpt/7B_rlhf/exp/"

# RL过程中Policy保存路径
POLICY_CKPT_PATH="/cognitive_comp/songzhuoyang/workspace/chatgpt/7B_rlhf/ckpt/$MODEL_NAME/"

# 初始Policy model的路径
POLICY_MODEL_PATH="/cognitive_comp/songzhuoyang/models/llama_sft/20230405v1/"

POLICY_TOKENIZER_PATH="/cognitive_comp/songzhuoyang/models/llama_sft/20230405v1/"

RM_CONFIG_PATH="/cognitive_comp/sunqianguo/pretrained/checkpoints/7B/0405/v2/checkpoint-16000/"
RM_MODEL_PATH="/cognitive_comp/liangyuxin/workspace/rm_train/ckpt_log/RM_0412_mix_7B/ckpt/last.ckpt/checkpoint/"

# sif镜像路径
SIF_FILE="/cognitive_comp/songzhuoyang/images/fengshen_chatgpt.sif"


EXPERIMENT_ARGS="\
    --wandb_project     $MODEL_NAME \
    --seed              42 \
    "

PPO_ARGS="\
    --num_episodes      256 \
    --max_timesteps     1 \
    --update_timesteps  1 \
    --sample_batch_size 32 \
    --buffer_limit_size 512 \
    --max_epoch_per_update  2 \
    --eps_clip          0.2 \
    --value_clip        0.2 \
    "

MEGATRON_ARGS="\
    --deepspeed_stage     1 \
    "

EXPERIENCE_ARGS="\
    --top_p                 0.85 \
    --top_k                 0 \
    --repetition_penalty    1. \
    --temperature           1 \
    --max_length            1024 \
    --kl_coef               0 \
    --experience_batch_size 32 \
    --policy_minibatch_size 4 \
    --rm_minibatch_size     1 \
    --prompt_dataset_path   /opt/prompt_dataset \
    --exp_save_path         /opt/exp \
    "

MODEL_ARGS="\
    --policy_ckpt_path          /opt/ckpt/ \
    --policy_tokenizer_path     /opt/policy_tokenizer/ \
    --policy_model_path         /opt/policy_model/ \
    --rm_config_path            /opt/rm_config/config.json\
    --rm_model_path             /opt/rm_model/mp_rank_00_model_states.pt \
    "

TRAINER_ARGS="\
    --do_validation     \
    --val_check_interval    5 \
    --val_size_per_task 20 \
    --policy_precision  bf16 \
    --learning_rate     5e-6 \
    --min_learning_rate 1e-7 \
    --total_steps       512 \
    --warmup_ratio      0.05 \
    --policy_train_batch_size   2 \
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


echo "START"

mkdir $POLICY_CKPT_PATH
mkdir $EXP_SAVE_PATH

singularity exec --nv \
    -B $CODE_PATH:/opt/code/ \
    -B $CHATGPT_PATH:/opt/chatgpt/chatgpt/ \
    -B $PROMPT_PATH:/opt/prompt_dataset/ \
    -B $EXP_SAVE_PATH:/opt/exp/ \
    -B $POLICY_CKPT_PATH:/opt/ckpt/ \
    -B $POLICY_TOKENIZER_PATH:/opt/policy_tokenizer/ \
    -B $POLICY_MODEL_PATH:/opt/policy_model/ \
    -B $RM_CONFIG_PATH:/opt/rm_config/ \
    -B $RM_MODEL_PATH:/opt/rm_model/ \
    $SIF_FILE python3 -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE /opt/code/$CODE_NAME $options
