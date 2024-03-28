
import torch
from fengshen_inner.models.megatron import mpu
from chatgpt.nn.llama import LlamaActor
from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM
from deepspeed.runtime.activation_checkpointing.checkpointing import _MODEL_PARALLEL_RNG_TRACKER_NAME
from transformers import AutoTokenizer
from chatgpt.nn.utils import zero_pad_sequences



mpu.set_model_parallel_world_size(1)
mpu.set_model_parallel_rank(0)
mpu.set_init_params_in_cuda(False)
mpu.get_cuda_rng_tracker().add(name=_MODEL_PARALLEL_RNG_TRACKER_NAME, seed=42)


# import random
# import numpy as np

# def set_seeds(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# # Set seeds for reproducibility
# set_seeds(42)

if __name__ == '__main__':
    model_path = '/cognitive_comp/wanghao/models/llama_sft/PPO_LLAMA13B_writing_RM13B_0707_step69_fs'
    # model_path = '/cognitive_comp/wanghao/models/llama_sft/PPO_LLAMA13B_writing_RM13B_0621_step74_fs'
    # model_path = '/cognitive_comp/wanghao/models/llama_sft/PPO_LLAMA13B_writing_RM13B_0621_step49_fs'
    actor = LlamaActor(
            model=LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16
            )
        ).half().cuda()

    tk_path = '/cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-c-pretrain-tokenizer/'
    llama_tokenizer = AutoTokenizer.from_pretrained(tk_path)
    generate_kwargs = {
            "do_sample": True,
            "top_p": 0.85,   
            "top_k": 0,
            "max_length": 2048,
            "repetition_penalty": 1.0,
            "temperature": 0.8,
            "use_cache": True,
            "pad_token_id": llama_tokenizer.eos_token_id,
            "eos_token_id": llama_tokenizer.eos_token_id,
    }

    queries = ['帮我写封信，给我喜欢的男明星，表达我特别喜欢他的影视作品，一定要凸显出他精致的五官帅到了我。',\
          '公司明天周六需要加班，请写一个书面通知，最好委婉点，同时给大家一些鼓励',\
          '如何撰写一份产品经理职位申请的求职信，并给出示例',\
          '我五一宅家五天，请写一个朋友圈文案让大家羡慕我',\
          '请提供一些有效的广告文案写作技巧，并在讲解完每个技巧后给出一个成功的广告示例',\
          '给电视剧《红楼梦》写一首歌，要求歌词能体现古时代女子的悲哀处境、人事复杂']
    texts = [f'<human>:{query}\n<bot>:' for query in queries]*5

    input_ids = [torch.tensor(llama_tokenizer(text).input_ids) for text in texts]
    input_ids = zero_pad_sequences(input_ids, side="right", padding_value=llama_tokenizer.eos_token_id)
    
    sequences = actor.generate(torch.tensor(input_ids).cuda(), **generate_kwargs)
    outputs = [llama_tokenizer.decode(seq.tolist()).replace('<s>','').replace('</s>', '') for seq in sequences[0]]
    print(outputs)

