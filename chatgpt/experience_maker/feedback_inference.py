from typing import Dict, List, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass, asdict
import gc
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from chatgpt.experience_maker import Experience, ExperienceMaker
from chatgpt.nn import Actor, Reflector
from chatgpt.nn.utils import zero_pad_sequences, shrink_mask
from chatgpt.utils import logging_rank_0, is_rank_0
from chatgpt.pipeline.tokenizer import FDPromptConvertion


@dataclass(kw_only=True)
class FDExperience:

    sequences: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    ref_sequences: torch.Tensor
    ref_attention_mask: torch.Tensor
    ref_action_mask: torch.Tensor
    scores: torch.Tensor
    repeating: torch.Tensor

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = self.sequences.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.action_mask = self.action_mask.to(device)
        self.ref_sequences = self.ref_sequences.to(device)
        self.ref_attention_mask = self.ref_attention_mask.to(device)
        self.ref_action_mask = self.ref_action_mask.to(device)
        self.scores = self.scores.to(device)
        self.repeating = self.repeating.to(device)
        return self

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        self.action_mask = self.action_mask.pin_memory()
        self.ref_sequences = self.ref_sequences.pin_memory()
        self.ref_attention_mask = self.ref_attention_mask.pin_memory()
        self.ref_action_mask = self.ref_action_mask.pin_memory()   
        self.scores = self.scores.pin_memory() 
        self.repeating = self.repeating.pin_memory() 
        return self
    
    @property
    def batchsize(self):
        return self.sequences.shape[0]


class FDExperienceMaker(ExperienceMaker):
    """
    Experience maker for feedback distillatiton.
    
    """

    def __init__(self,
        actor: Actor,
        reflector: Reflector,
        prompt_convertion: FDPromptConvertion,
        actor_minibatch_size: int = 16,
        reflector_minibatch_size: int = 16,
        actor_gen_args: Optional[dict] = None,
        reflector_gen_args: Optional[dict] = None,
        seed: int = 42,
        **kwargs) -> None:

        super().__init__(actor, None, None, None)
        self.reflector = reflector
        self.prompt_convertion = prompt_convertion
        self.actor_minibatch_size = actor_minibatch_size
        self.reflector_minibatch_size = reflector_minibatch_size

        self.actor_gen_args = actor_gen_args if actor_gen_args is not None else {}
        self.reflector_gen_args = reflector_gen_args if reflector_gen_args is not None else {}

        self.seed = seed
        torch.manual_seed(self.seed)

        return


    def _make_experience_with_actor(self, queries: List[str]) -> Dict[str, torch.Tensor]:

        minibatch_outputs = defaultdict(list)
        return_dict = {}
        try:
            device = self.actor.module.device
        except:
            device = self.actor.model.device

        with torch.no_grad():
            batch_size = len(queries)
            minibatch_size = self.actor_minibatch_size
            num_of_batch = batch_size // minibatch_size + bool(batch_size % minibatch_size)
            for i in tqdm(range(num_of_batch), desc='Making actor exps', disable=not is_rank_0()):
                minibatch_data = queries[i * minibatch_size: min(batch_size, (i + 1) * minibatch_size)]
                input_ids = self.prompt_convertion.to_actor(minibatch_data)
                seq, attn_msk, act_msk, _ = self.actor.module.generate(
                    input_ids=input_ids.to(device),
                    **self.actor_gen_args
                )          
                gc.collect()
                torch.cuda.empty_cache()
                minibatch_outputs['sequences'].append(seq)
                minibatch_outputs['attention_mask'].append(attn_msk)
                minibatch_outputs['action_mask'].append(act_msk)

            for key, value in minibatch_outputs.items():
                if key == "sequences":
                    return_dict[key] = zero_pad_sequences(value, 'right', self.actor_gen_args.get('pad_token_id', 0))
                else:
                    return_dict[key] = zero_pad_sequences(value, 'right', False)

        return return_dict
    

    def _make_experience_with_reflector(self, queries: List[str], response_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        minibatch_outputs = defaultdict(list)
        return_dict = {}
        try:
            device = self.reflector.model.device
        except:
            device = self.reflector.module.device

        with torch.no_grad():
            batch_size = len(queries)
            minibatch_size = self.reflector_minibatch_size
            num_of_batch = batch_size // minibatch_size + bool(batch_size % minibatch_size)
            for i in tqdm(range(num_of_batch), desc='Making reflector exps', disable=not is_rank_0()):
                minibatch_queries = queries[i * minibatch_size: min(batch_size, (i + 1) * minibatch_size)]
                minibatch_seq = response_dict['sequences'][i * minibatch_size: min(batch_size, (i + 1) * minibatch_size)]
                minibatch_act_mask = response_dict['action_mask'][i * minibatch_size: min(batch_size, (i + 1) * minibatch_size)]
                input_ids = self.prompt_convertion.to_reflector(minibatch_queries, minibatch_seq, minibatch_act_mask)
                feedback_seq, feedback_attn_mask, feedback_act_mask, _ = self.reflector.module.generate( 
                    input_ids=input_ids.to(device),
                    **self.reflector_gen_args
                )          
                gc.collect()
                torch.cuda.empty_cache()
                minibatch_outputs['sequences'].append(feedback_seq)
                minibatch_outputs['attention_mask'].append(feedback_attn_mask)
                minibatch_outputs['action_mask'].append(feedback_act_mask)

            for key, value in minibatch_outputs.items():
                if key == "sequences":
                    return_dict[key] = zero_pad_sequences(value, 'right', self.reflector_gen_args.get('pad_token_id', 0))
                else:
                    return_dict[key] = zero_pad_sequences(value, 'right', False)

        return return_dict


    def _prepare_refinement(self, queries, response_dict, feedback_dict) -> Dict[str, torch.Tensor]:
        
        pad_token_id = self.actor_gen_args.get('pad_token_id', 0)
        return_dict, scores, repeating = self.prompt_convertion.refine(queries, response_dict, feedback_dict, pad_token_id)

        return return_dict, scores, repeating


    @torch.no_grad()
    def make_experience(self, inputs: List[str]) -> FDExperience:

        self.reflector.eval()
        self.actor.eval()      
        response_dict = self._make_experience_with_actor(queries=inputs)
        gc.collect()
        
        # self.reflector.cuda()
        feedback_dict = self._make_experience_with_reflector(queries=inputs, response_dict=response_dict)
        # self.reflector.cpu()
        gc.collect()
    
        refinement, scores, repeating = self._prepare_refinement(inputs, response_dict, feedback_dict)
        experience = {
            'sequences': response_dict['sequences'],
            'attention_mask': response_dict['attention_mask'],
            'action_mask': response_dict['action_mask'],
            'ref_sequences': refinement['sequences'],
            'ref_attention_mask': refinement['attention_mask'],
            'ref_action_mask': refinement['action_mask'],
            'scores': torch.tensor(scores),
            'repeating': torch.tensor(repeating)
        }
        return FDExperience(**experience)
    

    def logging(self, text, level='info'):
        logging_rank_0(text, level=level)


@dataclass(kw_only=True)
class SandBoxExperience:

    train_basic_seq: torch.Tensor
    train_basic_att: torch.Tensor
    train_basic_act: torch.Tensor
    train_detail_seq: torch.Tensor
    train_detail_att: torch.Tensor
    train_detail_act: torch.Tensor
    train_pred: torch.Tensor
    train_ans: torch.Tensor
    test_seq: torch.Tensor
    test_att: torch.Tensor
    test_act: torch.Tensor
    test_ans: torch.Tensor

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.train_basic_seq = self.train_basic_seq.to(device)
        self.train_basic_att = self.train_basic_att.to(device)
        self.train_basic_act = self.train_basic_act.to(device)
        self.train_detail_seq = self.train_detail_seq.to(device)
        self.train_detail_att = self.train_detail_att.to(device)
        self.train_detail_act = self.train_detail_act.to(device)
        self.train_pred = self.train_pred.to(device)
        self.train_ans = self.train_ans.to(device)
        self.test_seq = self.test_seq.to(device)
        self.test_att = self.test_att.to(device)
        self.test_act = self.test_act.to(device)
        self.test_ans = self.test_ans.to(device)
        return self

    def pin_memory(self):
        self.train_basic_seq = self.train_basic_seq.pin_memory()
        self.train_basic_att = self.train_basic_att.pin_memory()
        self.train_basic_act = self.train_basic_act.pin_memory()
        self.train_detail_seq = self.train_detail_seq.pin_memory()
        self.train_detail_att = self.train_detail_att.pin_memory()
        self.train_detail_act = self.train_detail_act.pin_memory()
        self.train_pred = self.train_pred.pin_memory()
        self.train_ans = self.train_ans.pin_memory()
        self.test_seq = self.test_seq.pin_memory()
        self.test_att = self.test_att.pin_memory()
        self.test_act = self.test_act.pin_memory()
        self.test_ans = self.test_ans.pin_memory()
        return self
    
    @property
    def batchsize(self):
        return self.train_basic_seq.shape[0]


class SandBoxExperienceMaker(ExperienceMaker):

    def __init__(self, actor: Actor, tokenizer: Callable, template: str) -> None:
        super().__init__(None, None, None, None)
        self.actor = actor
        self.tokenizer = tokenizer
        assert '{__query__}' in template, "template should contains format field '{__query__}'"
        self.template = template

    def logging(self, text, level='info'):
        logging_rank_0(text, level=level)

    def make_experience(self, inputs: List[Dict]) -> SandBoxExperience:
        experience = defaultdict(list)
        try:
            device = self.actor.module.device
        except:
            device = self.actor.model.device

        for each in inputs:
            options_ids = [self.tokenizer(option, add_special_tokens=False).input_ids for option in each['train_options']]
            input_prefix_ids = self.tokenizer(self.template.format(__query__=' '.join([each['basic'], each['train_query']]))).input_ids
            input_ids = [torch.tensor(input_prefix_ids + option_ids)[None, :] for option_ids in options_ids]
            input_ids = zero_pad_sequences(input_ids, 'right', self.tokenizer.pad_token_id)
            att_msk = input_ids != self.tokenizer.pad_token_id
            act_msk = shrink_mask(att_msk, (len(input_prefix_ids), 0))[:, 1:]

            with torch.no_grad():
                log_prob = self.actor(input_ids.to(device), None, att_msk.to(device), return_logits=False)
                pred = (log_prob.cpu() * act_msk).sum(dim=-1).argmax().item()

            experience['train_basic_seq'].append(input_ids[pred][None, :])
            experience['train_basic_att'].append(att_msk[pred][None, :])
            experience['train_basic_act'].append(act_msk[pred][None, :])
            experience['train_pred'].append(pred)
            experience['train_ans'].append(each['train_answer'])

            input_prefix_ids = self.tokenizer(self.template.format(__query__=' '.join([each['basic'], each['details'], each['train_query']]))).input_ids
            train_detail_seq = torch.tensor(input_prefix_ids + options_ids[pred])[None, :]
            train_detail_att = torch.ones_like(train_detail_seq).bool()
            train_detail_act = shrink_mask(train_detail_att, (len(input_prefix_ids), 0))[:, 1:]

            experience['train_detail_seq'].append(train_detail_seq)
            experience['train_detail_att'].append(train_detail_att)
            experience['train_detail_act'].append(train_detail_act)    

            options_ids = [self.tokenizer(option, add_special_tokens=False).input_ids for option in each['test_options']]
            input_prefix_ids = self.tokenizer(self.template.format(__query__=' '.join([each['basic'], each['test_query']]))).input_ids
            input_ids = [torch.tensor(input_prefix_ids + option_ids)[None, :] for option_ids in options_ids]
            input_ids = zero_pad_sequences(input_ids, 'right', self.tokenizer.pad_token_id)
            att_msk = input_ids != self.tokenizer.pad_token_id
            act_msk = shrink_mask(att_msk, (len(input_prefix_ids), 0))[:, 1:]

            experience['test_seq'].append(input_ids[None, :])
            experience['test_att'].append(att_msk[None, :])
            experience['test_act'].append(act_msk[None, :])
            experience['test_ans'].append(each['test_answer'])

            gc.collect()
            torch.cuda.empty_cache()

        for k, v in experience.items():
            if 'ans' in k or 'pred' in k:
                experience[k] = torch.tensor(v)
            else:
                experience[k] = zero_pad_sequences(v, 'right', self.tokenizer.pad_token_id if 'seq' in k else False)

        return SandBoxExperience(**experience)


@dataclass(kw_only=True)
class SandBoxExperienceV2:

    sequences: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    ref_sequences: torch.Tensor
    ref_attention_mask: torch.Tensor
    ref_action_mask: torch.Tensor
    test_sequences: torch.Tensor
    test_attention_mask: torch.Tensor
    test_action_mask: torch.Tensor
    test_labels: torch.Tensor

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = self.sequences.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.action_mask = self.action_mask.to(device)
        self.ref_sequences = self.ref_sequences.to(device)
        self.ref_attention_mask = self.ref_attention_mask.to(device)
        self.ref_action_mask = self.ref_action_mask.to(device)
        self.test_sequences = self.test_sequences.to(device)
        self.test_attention_mask = self.test_attention_mask.to(device)
        self.test_action_mask = self.test_action_mask.to(device)
        self.test_labels = self.test_labels.to(device)
        return self

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        self.action_mask = self.action_mask.pin_memory()
        self.ref_sequences = self.ref_sequences.pin_memory()
        self.ref_attention_mask = self.ref_attention_mask.pin_memory()
        self.ref_action_mask = self.ref_action_mask.pin_memory()
        self.test_sequences = self.test_sequences.pin_memory()
        self.test_attention_mask = self.test_attention_mask.pin_memory()
        self.test_action_mask = self.test_action_mask.pin_memory()
        self.test_labels = self.test_labels.pin_memory()    
        return self    

    @property
    def batchsize(self):
        return self.sequences.shape[0]


class SandBoxExperienceMakerV2(ExperienceMaker):

    def __init__(self, 
        actor: Actor, 
        tokenizer: Callable, 
        template: str, 
        actor_gen_args: Optional[dict] = None, 
        seed: int = 42,
        **kwargs) -> None:

        super().__init__(actor, None, None, None)
        self.tokenizer = tokenizer
        assert '{__query__}' in template, "template should contains format field '{__query__}'"
        self.template = template

        self.gen_args = actor_gen_args if actor_gen_args is not None else {}
        self.seed = seed
        torch.manual_seed(self.seed)
        return 
    
    def logging(self, text, level='info'):
        logging_rank_0(text, level=level)

    def make_experience(self, inputs: List[Dict]) -> SandBoxExperienceV2:
        experience = defaultdict(list)
        try:
            device = self.actor.module.device
        except:
            device = self.actor.model.device
        self.actor.eval()

        with torch.no_grad():
            for each in inputs:
                query = ' '.join([each['basic'], each['train_query']])
                input_ids = self.tokenizer(self.template.format(__query__=query), return_tensors='pt').input_ids
                sequences, attention_mask, action_mask, _ = self.actor.module.generate(
                    input_ids=input_ids.to(device),
                    **self.gen_args
                )
                gc.collect()
                torch.cuda.empty_cache()

                ref_query = ' '.join([each['basic'], each['details'], each['train_query']])
                ref_input_ids = self.tokenizer(self.template.format(__query__=ref_query)).input_ids
                action_ids = torch.masked_select(sequences[:, 1:], action_mask).tolist()
                ref_sequences = torch.tensor(ref_input_ids + action_ids)[None, :]
                ref_attention_mask = torch.ones_like(ref_sequences).bool()
                ref_action_mask = torch.tensor([0] * (len(ref_input_ids) - 1) + [1] * len(action_ids))[None, :].bool()
                experience['sequences'].append(sequences.cpu())
                experience['attention_mask'].append(attention_mask.cpu())
                experience['action_mask'].append(action_mask.cpu())
                experience['ref_sequences'].append(ref_sequences)
                experience['ref_attention_mask'].append(ref_attention_mask)
                experience['ref_action_mask'].append(ref_action_mask)

                options_ids = [self.tokenizer(option, add_special_tokens=False).input_ids for option in each['test_options']]
                input_prefix_ids = self.tokenizer(self.template.format(__query__=' '.join([each['basic'], each['test_query']]))).input_ids
                input_ids = [torch.tensor(input_prefix_ids + option_ids)[None, :] for option_ids in options_ids]
                input_ids = zero_pad_sequences(input_ids, 'right', self.tokenizer.pad_token_id)
                att_msk = input_ids != self.tokenizer.pad_token_id
                act_msk = shrink_mask(att_msk, (len(input_prefix_ids), 0))[:, 1:]

                experience['test_sequences'].append(input_ids[None, :])
                experience['test_attention_mask'].append(att_msk[None, :])
                experience['test_action_mask'].append(act_msk[None, :])
                experience['test_labels'].append(each['test_answer'])

        for k, v in experience.items():
            if 'labels' in k:
                experience[k] = torch.tensor(v)
            else:
                experience[k] = zero_pad_sequences(v, 'right', self.tokenizer.pad_token_id if 'seq' in k else False)
        
        return SandBoxExperienceV2(**experience)