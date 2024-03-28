# encoding=utf-8
import os

import torch
import torch.distributed.run
from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM
from fengshen_inner.models.megatron import mpu
from tokenizers import AddedToken
from transformers import AutoTokenizer, LlamaTokenizer

from chatgpt.experience_maker import EDPOExperienceMaker
from chatgpt.logger import WandbLogger
from chatgpt.nn import GSArgs
from chatgpt.nn.llama import LlamaActor, modeling_fengshenLlama_rm
from chatgpt.nn.utils import zero_pad_sequences
from chatgpt.pipeline.config import EDPOPipelineConfig, ActorGranularity, RewardModelGranularity
from chatgpt.replay_buffer import (DistributedBatchSampler,
                                   PreferenceReplayBuffer)
from chatgpt.strategies import (build_deepspeed_config,
                                get_save_checkpoint_callback,
                                initialize_megatron, setup_model_and_optimizer)
from chatgpt.trainer import EDPOTrainer
from chatgpt.utils import local_rank, logging_rank_0

from .utils import concat_prompt, load_jsonline_data
from .tokenizer import GLUE, TOKENIZER

_SPECIAL_TOKENS_DICT = {'pad_token': '</s>'}
human = AddedToken("<human>",lstrip=False,rstrip=False,single_word=False,normalized=True)
bot = AddedToken("<bot>",lstrip=False,rstrip=False,single_word=False,normalized=True)
    

def get_dataloader_build_func(args: EDPOPipelineConfig, tokenizer:LlamaTokenizer):
    def build_dataloader(replay_buffer, episode, timestep, val_str=""):

        rank = mpu.get_data_parallel_rank()
        mp_rank = mpu.get_model_parallel_rank()
        pp_rank = mpu.get_pipe_parallel_rank()
        world_size = mpu.get_data_parallel_world_size()
        global_batch_size = args.policy_train_batch_size * world_size

        # if mp_rank == 0 and pp_rank == 0:
        #     hf_dataset.save_to_disk(os.path.join(args.exp_save_path, f"{args.wandb_project}{val_str}_ep{str(episode).zfill(3)}_{str(timestep).zfill(3)}_{rank}/"))

        # Use a simple sampler with distributed batch sampler
        sampler = torch.utils.data.SequentialSampler(replay_buffer)
        batch_sampler = DistributedBatchSampler(
            sampler=sampler,
            batch_size=global_batch_size,
            drop_last=True,
            rank=rank,
            world_size=world_size,
        )

        # Torch dataloader
        return torch.utils.data.DataLoader(
            replay_buffer,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=replay_buffer.collate_fn,
        )
    return build_dataloader
     

def load_dateset(args:EDPOPipelineConfig, tokenizer:LlamaTokenizer):
    from collections import defaultdict

    import datasets as ds
    ds.disable_caching()

    dataset = load_jsonline_data(
        path=args.dataset_path,
        prefix=args.prefix,
        seperator=args.multiturn_seperator
    )
    dataset = concat_prompt(
        dataset=dataset, prefix=args.prefix, seperator=args.multiturn_seperator
    )
    # 'task':[,,...], 'query':[,,...], 'preference':[[],[],...]
    # [{'input_ids':tensor, 'preference_sequences':tensor, 'task':str}]
    queries = list(dataset["query"])
    if "preference" not in dataset.column_names:
        preferences = []*len(queries)
    else:
        preferences = list(dataset["preference"])
        preferences = [[query+pref for pref in preferences[i]] for i,query in enumerate(queries)]

    samples = []
    for i in range(len(queries)):
        input_ids = tokenizer.encode(queries[i], return_tensors='pt').squeeze(0)
        if len(preferences[i]) > 0:
            preference_ids =  [tokenizer.encode(preference, return_tensors='pt').squeeze(0) for preference in preferences[i]]
            padded_pref_ids = zero_pad_sequences(preference_ids, side="right", padding_value=tokenizer.pad_token_id)
        else:
            padded_pref_ids = torch.tensor([[]])
        samples.append({'input_ids':input_ids, 'preference_sequences':padded_pref_ids})

    if "task" not in dataset.column_names or not args.do_validation:
        return samples, None
    
    tasks = list(dataset["task"])
    task_to_prompts = defaultdict(list)
    for task_name, query in zip(tasks, samples):
        task_to_prompts[task_name].append(query)
    
    train_inputs = []
    val_inputs = []
    for task_name, samples in task_to_prompts.items():
        samples = [{**sample, 'task':task_name} for sample in samples]
        if len(samples) <= args.val_size_per_task:
            train_inputs.extend(samples)

        # g = torch.Generator()                                                                           
        # g.manual_seed(args.seed + mpu.get_data_parallel_rank())                                                                       
        # random_idx = torch.randperm(len(samples), generator=g).tolist()
        # samples = [samples[i] for i in random_idx]

        # train_inputs.extend(samples[:-args.val_size_per_task])
        # val_inputs.extend(samples[-args.val_size_per_task:])
    
    val_inputs = None if len(val_inputs) == 0 else val_inputs
        
    return train_inputs, val_inputs

def launch_edpo(args:EDPOPipelineConfig):

    initialize_megatron(args)
    
    strategy = build_deepspeed_config(args)

    ### 准备用于保存的目录
    # root
    workspace_path = args.workspace_path
    os.makedirs(workspace_path, exist_ok=True)
    #exp
    # exp_save_path = os.path.join(workspace_path, f"exp")
    # os.makedirs(exp_save_path, exist_ok=True)
    # ckpt
    ckpt_save_path = os.path.join(workspace_path, f"ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    
    ### 初始化 logger
    logger = WandbLogger(
        project=args.wandb_project,
        group=args.wandb_group,
        entity=args.wandb_team,
        ignore=local_rank() != 0,
        name=f"edpo-{args.wandb_name}",
    )
    logger.log_hyperparams(args.__dict__)

    policy_tokenizer = TOKENIZER[args.policy_model_type](tokenizer_path=args.policy_tokenizer_path)

    ### 模型加载路径
    policy_model_path = os.path.join(workspace_path, f"models/policy")

    ### 加载模型
    policy_precision = torch.bfloat16 if args.policy_precision == "bf16" else torch.float16
    actor_granularity = ActorGranularity.sample
    rm_granularity = RewardModelGranularity[args.rm_granularity] if args.rm_granularity else RewardModelGranularity.sample

    if args.has_ref_model_constraints:
        logging_rank_0(f"Use initial model as constrain.")
        im = LlamaActor(
            model=LlamaForCausalLM.from_pretrained(
                f"{policy_model_path}/part_{mpu.get_model_parallel_rank()}",
                torch_dtype=policy_precision
            ),
            actor_granularity=actor_granularity,
            rm_granularity=rm_granularity
        ).eval().to(dtype=policy_precision).cpu() #.to(dtype=torch.bfloat16)
    else:
        im = None
        
    model_glue = GLUE.get(f"{args.policy_model_type}_to_{args.reward_model_type}", None)
    glue = None if model_glue is None else model_glue(args=args, src_tokenizer=policy_tokenizer, dst_tokenizer=rm_tokenizer)
    
    # reward model
    # 如果启动了 equalizing_preferences，enabling_bon 或者 enabling_tot，需要使用 RM
    if args.equalizing_preferences or args.enabling_bon or args.enabling_tot:
        logging_rank_0(f"Use reward model.")
        rm_model_path = os.path.join(workspace_path, "models/reward_model")
        rm_precision = torch.bfloat16 if args.rm_precision == "bf16" else torch.float16
        try:
            rm_tokenizer = TOKENIZER[args.reward_model_type](tokenizer_path=args.rm_tokenizer_path)
            reward_model = modeling_fengshenLlama_rm(
                pretrained_path=f"{rm_model_path}/part_{mpu.get_model_parallel_rank()}",
                convert_func=glue,
                actor_granularity=actor_granularity,
                rm_granularity=rm_granularity,
                logger=logger,
            ).to(dtype=rm_precision).cpu()
        except:
            logging_rank_0(f"Fail to load reward model. Disable it.")
            args.equalizing_preferences = False
            args.enabling_bon = False
            args.enabling_tot = False
            reward_model = None
            rm_tokenizer = None
    else:
        reward_model, rm_tokenizer = None, None

    actor_model = LlamaForCausalLM.from_pretrained(
            f"{policy_model_path}/part_{mpu.get_model_parallel_rank()}",
            torch_dtype=policy_precision
        )
    if args.activation_checkpointing:                                                                                      
        actor_model.gradient_checkpointing_enable()
    actor = LlamaActor(model=actor_model, actor_granularity=actor_granularity, rm_granularity=rm_granularity).to(dtype=policy_precision)
    
 
    actor, actor_optimizer, actor_lr = setup_model_and_optimizer(args, actor, strategy)
    
    tokenizer_vocab_size = policy_tokenizer.vocab_size
    policy_vocab_size = im.model.config.vocab_size

    bad_words_ids = [[policy_tokenizer.convert_tokens_to_ids(role)] for role in ['<human>','<bot>']]
    logging_rank_0(f"Ignore all role tokens: bad_words_ids", 'debug')

    if policy_vocab_size > tokenizer_vocab_size:
        bad_words_ids += [[ids] for ids in range(tokenizer_vocab_size, policy_vocab_size)]
        logging_rank_0(f"BAD TOKEN IDS: {tokenizer_vocab_size}~{policy_vocab_size - 1}", 'debug')

    # 初始化trainer
    generate_kwargs = {
        "do_sample": True,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "bad_words_ids": bad_words_ids,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "use_cache": True,
        "pad_token_id": policy_tokenizer.pad_token_id,
        "eos_token_id": policy_tokenizer.eos_token_id,
    }

    # 初始化experience_maker replay_buffer
    experience_maker = EDPOExperienceMaker(
        actor=actor,
        initial_model=im,
        reward_model=reward_model,
        seed=args.seed + mpu.get_data_parallel_rank(),
        pad_token_id=policy_tokenizer.pad_token_id,
        eos_token_id=policy_tokenizer.eos_token_id,
        gen_minibatch_size=args.generate_minibatch_size,
        actor_minibatch_size=args.policy_minibatch_size,
        rm_minibatch_size=args.rm_minibatch_size,
        gen_args=generate_kwargs,
        gs_args=GSArgs(enabling_tot=args.enabling_tot,
                          gs_eval_batch_size=args.gs_eval_batch_size,
                          gs_gen_batch_size=args.gs_gen_batch_size,
                          gs_gen_repeat_times=args.gs_gen_repeat_times,
                          gs_breadth=args.gs_breadth,
                          gs_iterations=args.gs_iterations,
                          generator_tokenizer=policy_tokenizer,
                          evaluator_tokenizer=rm_tokenizer),
        equalizing_preferences=args.equalizing_preferences,
        max_n_preferences=args.max_n_preferences)


    replay_buffer = PreferenceReplayBuffer(
        sample_batch_size=args.sample_batch_size,
        limit=args.buffer_limit_size,
        cpu_offload=args.replay_buffer_cpu_offload,
        pad_token_id=policy_tokenizer.pad_token_id
    )

    dpo_trainer = EDPOTrainer(
        actor=actor,
        actor_optim=actor_optimizer,
        actor_lr_scheduler = actor_lr,
        experience_maker=experience_maker,
        replay_buffer=replay_buffer,
        beta=args.dpo_beta,
        logger=logger,
        clip_grad=args.clip_grad,
        ckpt_saving_func=get_save_checkpoint_callback(path=ckpt_save_path, save_optimizer=False),
        experience_batch_size=args.experience_batch_size,
        setup_dataloader_func=get_dataloader_build_func(args, policy_tokenizer),
        max_epochs=args.max_epoch_per_update,                  # 每个训练阶段actor和critic的训练轮数
        tokenizer=policy_tokenizer,
        sample_replay_buffer=args.sample_replay_buffer,   # 每次使用全部经验池内容
        **generate_kwargs           # 模型生成样本的参数
    )

    train_inputs, val_inputs = load_dateset(args, policy_tokenizer)
    
    if val_inputs is not None:
        logging_rank_0(f"Train on {len(train_inputs)} prompts with {len(val_inputs)} validation every {args.val_check_interval} episode.", "debug")
    else:
        logging_rank_0(f"Train on {len(train_inputs)} prompts without validation.", "debug")

    # 开始训练
    dpo_trainer.fit(
        inputs=train_inputs,
        val_inputs=val_inputs,
        seed=args.seed + mpu.get_data_parallel_rank(),
        val_check_interval=args.val_every_n_episode,
        save_every_n_episode=args.save_every_n_episode,
        num_episodes=args.num_episodes,
        max_timesteps=args.max_timesteps,     # 每一个episode采样经验的步数
        ignore_ref_first_n_steps=args.ignore_ref_first_n_steps,
        update_timesteps=args.update_timesteps,   # 训练模型的累积步数
    )
    return
