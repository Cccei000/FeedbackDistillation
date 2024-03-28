
import torch
from fengshen_inner.models.megatron import mpu
from chatgpt.nn.llama import LlamaActor
from fengshen_inner.models.llama.modeling_llama import LlamaForCausalLM
from chatgpt.nn.llama import modeling_llama_rm
from deepspeed.runtime.activation_checkpointing.checkpointing import _MODEL_PARALLEL_RNG_TRACKER_NAME
from transformers import AutoTokenizer

from transformers import AutoTokenizer
import torch.nn.functional as F
import re
from chatgpt.nn.utils import zero_pad_sequences

_SEED = 42
mpu.set_model_parallel_world_size(1)
mpu.set_model_parallel_rank(0)
mpu.set_init_params_in_cuda(False)
mpu.get_cuda_rng_tracker().add(name=_MODEL_PARALLEL_RNG_TRACKER_NAME, seed=_SEED)

import random
import numpy as np
import pandas as pd

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set seeds for reproducibility

class Evaluator:
    def __init__(self, model, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def calculate(self, texts):
        with torch.no_grad():
            input_ids = [torch.tensor(self.tokenizer(text).input_ids) for text in texts]
            padded_input_ids = zero_pad_sequences(input_ids, side="right", padding_value=self.tokenizer.eos_token_id)
            padded_input_ids = padded_input_ids.to(self.device)
            attention_mask = padded_input_ids.not_equal(self.tokenizer.eos_token_id)

            scores = self.model(padded_input_ids, attention_mask=attention_mask)

        return scores.detach().tolist()
    
class Generator:
    def __init__(self, model, tokenizer: AutoTokenizer, kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.kwargs = kwargs
    
    def generate(self, inputs, split_by_punc=True):
        '''
        '''
        # print(self.kwargs)
        pad_token_id = self.kwargs["pad_token_id"]
        with torch.no_grad():
            outputs = []
            input_ids = [torch.tensor(self.tokenizer(prompt).input_ids) for prompt in inputs]
            padded_input_ids = zero_pad_sequences(input_ids, side="right", padding_value=pad_token_id)

            sequences = self.model.generate(
                padded_input_ids.to(self.device), **self.kwargs)[0]
            for i,seq in enumerate(sequences.detach().tolist()):
                output = []
                if split_by_punc:
                    out_text = self.tokenizer.decode(seq[len(input_ids[i]):], skip_special_tokens=False)
                    out_text = self.rsplit_on_last_dot_or_question(out_text)
                else:
                    out_text = self.tokenizer.decode(seq, skip_special_tokens=False).replace('<bot> :', '<bot>:').split('<bot>:')[-1]
                output.append(out_text.replace('<s>','').replace('</s>','').strip())
                outputs.extend(output)
        return outputs
    
    def rsplit_on_last_dot_or_question(self, s):
        splitted = re.split(r'[。|？|!|\n]', s)
        if len(splitted) == 1:
            splitted = re.split(r'[，｜,]', s)
        if len(splitted) == 1:
            splitted = re.split(r'[.]', s)
        if len(splitted) > 1:
            return s[:len(s)-len(splitted[-1])]
        else:
            return s


class TreeNode:
    def __init__(self, sentence, score, parent=None, generator=None, scorer=None):
        self.sentence = sentence
        self.score = score
        self.parent = parent
        self.children = []
        self.generator = generator
        self.scorer = scorer

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        if self.parent is None:
            return f"Sentence: {self.sentence.replace('<human>:','').replace('<bot>:', '')}"
        else:
            return f"Sentence: {self.sentence} Score: {self.score}"

def generate_candidates(node, num_cand=3):
    # 获取完整句子
    if node is None:
        return [('',0.0)]
    cur_node = node
    prompt = cur_node.sentence
    while cur_node.parent is not None:
        prompt = cur_node.parent.sentence + prompt
        cur_node = cur_node.parent 
    prompt_list = [prompt]*num_cand
    cand_sentences = generator.generate(prompt_list, split_by_punc=True)
    prompt_list_rm =  [prompt+cand_i for cand_i in cand_sentences]
    cand_scores = scorer.calculate(prompt_list_rm)
    candidates = [(sent, score) for sent,score in zip(cand_sentences, cand_scores)]
    return candidates


def build_tree(root, depth):
    if depth == 0:
        return
    candidates = generate_candidates(root)
    for sentence, score in candidates:
        child = TreeNode(sentence, score, root)
        root.add_child(child)
        build_tree(child, depth - 1)

def print_tree(root, str_all, str_prefix):
    str_curr = f"Sentence: {root.sentence} Score: {root.score}"
    str_all = str_all + str_prefix + str_curr + "\n" 
    if len(root.children) == 0:
        print(str_all)
        return
    for child in root.children:
        print_tree(child, str_all, str_prefix+"    ")
    


if __name__ == '__main__':
    model_path = '/cognitive_comp/wanghao/models/llama_sft/merged_stage2.2step6600_stage3step11800_stage3.1step7000_fs'
    actor = LlamaActor(
            model=LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16
            )
        ).to(dtype=torch.bfloat16).cuda()

    tk_path = '/cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-c-pretrain-tokenizer/'
    llama_tokenizer = AutoTokenizer.from_pretrained(tk_path)

    # rm_model_path = '/cognitive_comp/liangyuxin/workspace/chatgpt/13B_RM/ckpt/RM_LLAMA_13B_0612/global_step32383_hf'
    rm_model_path = '/cognitive_comp/shenjunyi/g/chatgpt/chatgpt/workspace/ckpt/FG_RM_LLAMA_13B_0621_math_zh_en/global_step36570_hf'
    rm_model = modeling_llama_rm(rm_model_path).half().cuda()

    generate_kwargs = {
        "do_sample": True,
        "top_p": 0.85,   
        "top_k": 0,
        "max_new_tokens": 50,
        "repetition_penalty": 1.0,
        "temperature": 0.8,
        "use_cache": True,
        "pad_token_id": llama_tokenizer.eos_token_id,
        "eos_token_id": llama_tokenizer.eos_token_id,
    }

df = pd.read_excel("/cognitive_comp/liangyuxin/datasets/chatgpt/human_eval/all_eval_results_with_reward.xlsx")
df_math = df[df["category"]=="数学题"]
querys = df_math['query'].to_list()

prefix_user = "<human>:"
prefix_bot = "\n<bot>:"
for query_i in querys:
    set_seeds(_SEED)
    scorer = Evaluator(rm_model, llama_tokenizer)
    generator = Generator(actor, llama_tokenizer, generate_kwargs)
    root_sentence = prefix_user+query_i+prefix_bot
    # root_sentence = '<human>:求解方程 "2x + 7 = 15\n<bot>:'
    root_score = 0.0
    root = TreeNode(root_sentence, root_score)

    # 设定树的深度，这里用3作为例子
    depth = 3
    build_tree(root, depth)

    # 打印树结构
    #
    print(root)
    for child in root.children:
        print('  ', child)
        for grandchild in child.children:
            print('    ', grandchild)

    print("-----------------------------")
    
    print_tree(root, "", "")
