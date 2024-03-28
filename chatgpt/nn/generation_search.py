# encoding=utf-8
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import re
from chatgpt.nn.utils import zero_pad_sequences, pad_2d_tensors, concat_2d_tensors
from torch.nn.utils.rnn import pad_sequence
import heapq
from abc import ABC
from chatgpt.utils import logging_rank_0
from dataclasses import dataclass


@dataclass
class GSArgs:
    enabling_tot:           bool = False
    gs_eval_batch_size:      int = 1
    gs_gen_batch_size:       int = 1
    enabling_bon:           bool = False
    best_of_n_times:         int = 1
    min_step_lengths:        int = 1
    gs_gen_repeat_times:     int = 1
    gs_breadth:              int = 1
    gs_iterations:           int = 1
    generator_tokenizer:  AutoTokenizer = None
    evaluator_tokenizer:  AutoTokenizer = None


def llama_decoding_post_processing(text):
    text = text.replace('<bot> :', '<bot>:').replace('<human> :', '<human>:')
    text = text.split('<bot>:')[-1]
    text = text.replace('<s>','').replace('</s>','')
    return text

def rsplit_on_last_dot_or_question(s):
    splitted = re.split(r'[。|？|!|\n]', s)
    if len(splitted) == 1:
        splitted = re.split(r'[，｜,]', s)
    if len(splitted) == 1:
        splitted = re.split(r'[.]', s)
    if len(splitted) > 1:
        s = s[:len(s)-len(splitted[-1])]
    s = s.replace('<bot> :', '<bot>:').replace('<human> :', '<human>:').replace('<s>','').replace('</s>','')
    return s

class Evaluator:
    def __init__(self, model, tokenizer: AutoTokenizer, bs: int=1):
        self.model = model
        self.tokenizer = tokenizer
        self.bs = bs
        self.device = next(model.parameters()).device

    def calculate(self, texts):
        with torch.no_grad():
            outputs = []
            for i in range(0, len(texts), self.bs):
                full_texts = texts[i:i+self.bs]
                input_ids = [torch.tensor(self.tokenizer(text).input_ids) for text in full_texts]
                padded_input_ids = zero_pad_sequences(input_ids, side="right", padding_value=self.tokenizer.eos_token_id)
                padded_input_ids = padded_input_ids.to(self.device)
                attention_mask = (padded_input_ids == self.tokenizer.eos_token_id).cumsum(dim=-1) <= 1
                scores = self.model(padded_input_ids, attention_mask=attention_mask)
                outputs.extend(scores.detach().tolist())

        return outputs
class Generator:
    def __init__(self, model, tokenizer: AutoTokenizer, bs: int=1, kwargs=None):
        self.model = model
        self.tokenizer = tokenizer
        self.bs = bs
        self.device = next(model.parameters()).device
        self.kwargs = kwargs
        logging_rank_0(f"{kwargs}")

    def generate(self, inputs, split_by_punc=True):
        '''
        '''
        pad_token_id = self.kwargs["pad_token_id"]
        with torch.no_grad():
            outputs = []
            for i in range(0, len(inputs), self.bs):
                prompts = inputs[i:(i+self.bs)]
                input_ids = [torch.tensor(self.tokenizer(prompt).input_ids) for prompt in prompts]
                padded_input_ids = zero_pad_sequences(input_ids, side="right", padding_value=pad_token_id)

                sequences = self.model.generate(
                    padded_input_ids.to(self.device), **self.kwargs)
                
                # hack后的hf generate会返回生成效率信息，需要适配
                if isinstance(sequences, tuple):
                    sequences = sequences[0]

                output = []
                for j,seq in enumerate(sequences.detach().tolist()):
                    if split_by_punc:
                        out_text = self.tokenizer.decode(seq[len(input_ids[j]):], skip_special_tokens=False)
                        out_text = rsplit_on_last_dot_or_question(out_text)
                    else:
                        out_text = self.tokenizer.decode(seq, skip_special_tokens=False)
                        out_text = llama_decoding_post_processing(out_text)
                    output.append(out_text)
                outputs.extend(output)

        return outputs


class TreeNode:
    def __init__(self, sentence, score, parent=None):
        self.sentence = sentence
        self.score = score
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        if self.parent is None:
            return f"Sentence: {self.sentence.replace('<human>:','').replace('<bot>:', '')}"
        else:
            return f"Sentence: {self.sentence} Score: {self.score}"

class TotGS(ABC):
    def __init__(self, actor, reward_model: nn.Module, 
                actor_tokenizer: AutoTokenizer, rm_tokenizer: AutoTokenizer, 
                actor_batch_size=1, rm_batch_size=1,
                **generate_kwargs):
        super().__init__()
        self.scorer = Evaluator(reward_model, rm_tokenizer, bs=rm_batch_size)
        self.generator = Generator(actor, actor_tokenizer, bs=actor_batch_size, kwargs=generate_kwargs)

    def init_trees(self, input_ids: torch.Tensor):
        x = [self.generator.tokenizer.decode(ids.tolist()).replace('<s>','').replace('</s>', '').replace('<bot> :', '<bot>:').replace('<human> :', '<human>:') for ids in input_ids]
        self.generator.kwargs['max_new_tokens'] = 50

        root_nodes = [TreeNode(xi, 0.0) for xi in x]
        return [root_nodes]

    @classmethod
    def get_full_contexts(cls, nodes):
        prompts = ['']*len(nodes)
        for i,node in enumerate(nodes):
            if node is not None:
                cur_node = node
                prompt = cur_node.sentence
                while cur_node.parent is not None:
                    prompt = cur_node.parent.sentence + prompt
                    cur_node = cur_node.parent 
                prompts[i] = prompt
        return prompts

    def generate_candidates(self, nodes, k=3):
        # 获取完整句子
        bs = len(nodes)
        contexts = self.get_full_contexts(nodes)
        contexts = contexts * k
        cand_sentences = self.generator.generate(contexts, split_by_punc=True) # .[x1,x2,...xbs,x1,x2,...xbs,x1,x2,...xbs,...] repeat k times
        cand_scores = self.scorer.calculate([contexts[i]+cand_sentences[i] for i in range(len(contexts))])

        candidates = [
            [(cand_sentences[j * bs + i], cand_scores[j * bs + i]) 
                for j in range(k)
            ]
            for i in range(bs)
        ] # [[x1,x1,x1,...], [x2,x2,x2,...],...]

        return candidates

    def thought_generator(self, nodes, k=1):
        # 实现根据当前状态生成下一句话的函数
        self.generator.kwargs['max_new_tokens'] = 512
        contexts = self.get_full_contexts(nodes)
        output_texts = self.generator.generate(contexts, split_by_punc=False)
        # sequences_tensor = pad_2d_tensors(sequences, padding_value=self.generator.kwargs['pad_token_id'])
        output_sequences = pad_sequence(
            [torch.tensor(self.generator.tokenizer(item).input_ids[1:]) for item in output_texts], batch_first=True, padding_value=self.generator.kwargs['pad_token_id']
        ) # remove bos

        return output_sequences 

    def search(self, input_ids: torch.Tensor, T=1, k=2, b=2):
    #     #Tree-of-Thought: breadth-first-search
    #     #param x: inputs
    #     #param T: repeat generation times
    #     #param k: search steps
    #     #param b: search breadth 

        queue_new = self.init_trees(input_ids)
        bs = input_ids.shape[0]

        global_next_levels = [[]]*bs
        for t in range(1, T + 1):
            queue = queue_new
            for nodes in queue:
                next_levels = []
                candidates = self.generate_candidates(nodes, k)
                for i in range(bs):
                    child_nodes = [TreeNode(s_prime, score, parent=nodes[i]) for s_prime,score in candidates[i]]
                    [nodes[i].add_child(child_node) for child_node in child_nodes]
                    next_levels.append(child_nodes)
                
                global_next_levels = [g+l for g,l in zip(global_next_levels, next_levels)]
                next_level_scores = [[node.score for node in nodes] for nodes in global_next_levels]

                top_b_indices = [heapq.nlargest(b, range(len(next_level_score)), key=lambda i: next_level_score[i]) \
                            for next_level_score in next_level_scores]
                queue_new = [
                                [global_next_levels[i][top_b_indices[i][j]]
                                    for i in range(bs)
                                ]
                                for j in range(b)
                ]

        max_nodes = queue_new[0]
        logging_rank_0(f'Rank-{input_ids.device}-TOT search: current top queue {queue_new}')

        output_sequences = self.thought_generator(max_nodes, 1)

        concatenated_tensor = concat_2d_tensors([input_ids, output_sequences.to(input_ids.device)], padding_value=self.generator.kwargs['pad_token_id'])

        return concatenated_tensor, output_sequences


'''
if __name__ == '__main__':    
    x = ["写一首童话故事", "预测下明天的股市"]
    k = 5
    T = 5
    b = 2
    generate_kwargs = {
            "do_sample": True,
            "top_p": 0.85,   
            "top_k": 0,
            "max_new_tokens": 50,
            "repetition_penalty": 1.0,
            "temperature": 0.8,
            "pad_token_id": llama_tokenizer.eos_token_id,
            "eos_token_id": llama_tokenizer.eos_token_id,
    }

    args = {
        "gs_eval_batch_size": 5,
        "gs_gen_batch_size": 5,
        "gs_breadth": 5,
        "gs_iterations": 5,
        "gs_gen_repeat_times": 2,
    }

    scorer = Evaluator(args, rm_model, llama_tokenizer)
    generator = Generator(model, llama_tokenizer, args, generate_kwargs)
    result = tot_bfs(x, generator, scorer, args)
    print(result)
'''
