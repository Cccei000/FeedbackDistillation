
GPUS_PER_NODE=8

MODEL_NAME=PPO_NEOX_0414
CODE_PATH='/cognitive_comp/songzhuoyang/projects/chatgpt/chatgpt/examples/train_neox_ppo.py'

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=53005

# export LAUNCHER="torchrun \
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT \
#     --max_restarts 0 \
#     "

EXPERIMENT_ARGS="\
    --wandb_project        $MODEL_NAME \
    "

PPO_ARGS="\
    --num_episodes      1024 \
    --max_timesteps     1 \
    --update_timesteps  1 \
    --sample_batch_size 32 \
    --buffer_limit_size 512 \
    --max_epoch_per_update  2 \
    --eps_clip          0.2 \
    --value_clip        0.2 \
    "

EXPERIENCE_ARGS="\
    --top_p             0.7 \
    --top_k             0 \
    --repetition_penalty 1. \
    --temperature       1. \
    --max_length        1024 \
    --kl_coef           0 \
    --experience_batch_size 32 \
    --policy_minibatch_size 4 \
    --rm_minibatch_size 1 \
    --prompt_dataset_path /cognitive_comp/songzhuoyang/processed_data/mixed_ppo_dataset_0327_6b \
    --exp_save_path    /cognitive_comp/songzhuoyang/workspace/chatgpt/6B_rlhf/exp \
    "

MODEL_ARGS="\
    --policy_ckpt_path  /cognitive_comp/songzhuoyang/workspace/chatgpt/6B_rlhf/ckpt \
    --policy_tokenizer_path /cognitive_comp/songzhuoyang/models/neox_sft/20230324 \
    --policy_model_path /cognitive_comp/songzhuoyang/models/neox_sft/20230324 \
    "

TRAINER_ARGS="\
    --learning_rate     1e-6 \
    --min_learning_rate 1e-7 \
    --total_steps       512 \
    --warmup_ratio      0.02 \
    --policy_train_batch_size   2 \
    --scheduler_type    cosine \
    "

export options=" \
    $EXPERIMENT_ARGS \
    $PPO_ARGS \
    $EXPERIENCE_ARGS \
    $MODEL_ARGS \
    $TRAINER_ARGS \
    "

echo "START"
# $LAUNCHER --node_rank 0 $RUN_CMD
python3 -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE $CODE_PATH $options
