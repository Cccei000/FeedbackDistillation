
GPUS_PER_NODE=1

MODEL_NAME=PPO_ChatGLM_0420_1024
CODE_PATH='/cognitive_comp/wanghao/experiments/chatgpt/chatgpt/examples/train_chatglm_ppo.py'

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=53013

# export LAUNCHER="torchrun \
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT \
#     --max_restarts 0 \
#     "

EXPERIMENT_ARGS="\
    --wandb_project     $MODEL_NAME \
    --seed              42 \
    "

PPO_ARGS="\
    --num_episodes      256 \
    --max_timesteps     1 \
    --update_timesteps  1 \
    --sample_batch_size 128 \
    --buffer_limit_size 512 \
    --max_epoch_per_update  2 \
    --eps_clip          0.2 \
    --value_clip        0.2 \
    "

EXPERIENCE_ARGS="\
    --top_p             0.95 \
    --top_k             0 \
    --repetition_penalty 1. \
    --temperature       1 \
    --max_length        1024 \
    --kl_coef           0 \
    --experience_batch_size 128 \
    --policy_minibatch_size 4 \
    --rm_minibatch_size 1 \
    --prompt_dataset_path /cognitive_comp/songzhuoyang/processed_data/mixed_ppo_dataset_0327_for_chatglm \
    --exp_save_path    /cognitive_comp/wanghao/experiments/workspace/chatgpt/chatglm_rlhf/exp \
    "

MODEL_ARGS="\
    --policy_ckpt_path  /cognitive_comp/wanghao/experiments/workspace/chatgpt/chatglm_rlhf/ckpt/$MODEL_NAME/ \
    --policy_tokenizer_path /cognitive_comp/wanghao/models/chatglm_6b \
    --policy_model_path /cognitive_comp/wanghao/models/chatglm_6b \
    --rm_config_path /cognitive_comp/sunqianguo/pretrained/checkpoints/7B/0405/v2/checkpoint-16000/config.json \
    --rm_model_path /cognitive_comp/liangyuxin/workspace/rm_train/ckpt_log/RM_0412_mix_7B/ckpt/last.ckpt/checkpoint/mp_rank_00_model_states.pt \
    --rm_model_tokenizer_path /cognitive_comp/songzhuoyang/models/llama_sft/20230405v1 \
    "

TRAINER_ARGS="\
    --policy_precision  fp16 \
    --learning_rate     5e-6 \
    --min_learning_rate 1e-7 \
    --total_steps       512 \
    --warmup_ratio      0.05 \
    --policy_train_batch_size   1 \
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
