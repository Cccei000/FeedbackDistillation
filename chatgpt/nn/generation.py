import gc
import time
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from chatgpt.utils import logging_rank_0, FlopsTimer

try:
    from transformers.generation_logits_process import (
        LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)
except ImportError:
    from transformers.generation import (LogitsProcessorList,
                                         RepetitionPenaltyLogitsProcessor,
                                         TemperatureLogitsWarper,
                                         TopKLogitsWarper, TopPLogitsWarper)
    from transformers.generation.utils import logger
    
    
    logger.disabled = True


def prepare_logits_processor(top_k: Optional[int] = None,
                             top_p: Optional[float] = None,
                             temperature: Optional[float] = None,
                             repetition_penalty: Optional[float] = 1.0,) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if repetition_penalty is not None and repetition_penalty != 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if temperature is not None and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        processor_list.append(TopKLogitsWarper(top_k))
    if top_p is not None and top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    return processor_list


def sample(model: nn.Module,
           input_ids: torch.Tensor,
           max_length: int,
           early_stopping: bool = False,
           eos_token_id: Optional[int] = None,
           pad_token_id: Optional[int] = None,
           top_k: Optional[int] = None,
           top_p: Optional[float] = None,
           temperature: Optional[float] = None,
           prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
           update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
           repetition_penalty: Optional[float] = 1.0,
           **model_kwargs) -> torch.Tensor:
    if input_ids.size(1) >= max_length:
        return input_ids

    logits_processor = prepare_logits_processor(top_k, top_p, temperature, repetition_penalty)
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

    for _ in range(input_ids.size(1), max_length):
        model_inputs = prepare_inputs_fn(input_ids, **model_kwargs) if prepare_inputs_fn is not None else {
            'input_ids': input_ids
        }
        outputs = model(**model_inputs)

        next_token_logits = outputs['logits'][:, -1, :]
        # pre-process distribution
        next_token_logits = logits_processor(input_ids, next_token_logits)
        # sample
        probs = torch.softmax(next_token_logits, dim=-1, dtype=torch.float)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if update_model_kwargs_fn is not None:
            model_kwargs = update_model_kwargs_fn(outputs, **model_kwargs)

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        # stop when each sentence is finished if early_stopping=True
        if early_stopping and unfinished_sequences.max() == 0:
            break

    return input_ids

def generate(model: nn.Module,
             input_ids: torch.Tensor,
             max_length: int,
             num_beams: int = 1,
             do_sample: bool = True,
             early_stopping: bool = False,
             eos_token_id: Optional[int] = None,
             pad_token_id: Optional[int] = None,
             top_k: Optional[int] = None,
             top_p: Optional[float] = None,
             temperature: Optional[float] = None,
             prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
             update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
             repetition_penalty: Optional[float] = 1.,
             **model_kwargs) -> torch.Tensor:
    """Generate token sequence. The returned sequence is input_ids + generated_tokens.

    Args:
        model (nn.Module): model
        input_ids (torch.Tensor): input sequence
        max_length (int): max length of the returned sequence
        num_beams (int, optional): number of beams. Defaults to 1.
        do_sample (bool, optional): whether to do sample. Defaults to True.
        early_stopping (bool, optional): if True, the sequence length may be smaller than max_length due to finding eos. Defaults to False.
        eos_token_id (Optional[int], optional): end of sequence token id. Defaults to None.
        pad_token_id (Optional[int], optional): pad token id. Defaults to None.
        top_k (Optional[int], optional): the number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to None.
        top_p (Optional[float], optional): If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. Defaults to None.
        temperature (Optional[float], optional): The value used to module the next token probabilities. Defaults to None.
        prepare_inputs_fn (Optional[Callable[[torch.Tensor, Any], dict]], optional): Function to preprocess model inputs. Arguments of this function should be input_ids and model_kwargs. Defaults to None.
        update_model_kwargs_fn (Optional[Callable[[dict, Any], dict]], optional): Function to update model_kwargs based on outputs. Arguments of this function should be outputs and model_kwargs. Defaults to None.
    """
    is_greedy_gen_mode = ((num_beams == 1) and do_sample is False)
    is_sample_gen_mode = ((num_beams == 1) and do_sample is True)
    is_beam_gen_mode = ((num_beams > 1) and do_sample is False)
    if is_greedy_gen_mode:
        # run greedy search
        raise NotImplementedError
    elif is_sample_gen_mode:
        # run sample
        return sample(model,
                      input_ids,
                      max_length,
                      early_stopping=early_stopping,
                      eos_token_id=eos_token_id,
                      pad_token_id=pad_token_id,
                      top_k=top_k,
                      top_p=top_p,
                      temperature=temperature,
                      prepare_inputs_fn=prepare_inputs_fn,
                      repetition_penalty=repetition_penalty,
                      update_model_kwargs_fn=update_model_kwargs_fn,
                      **model_kwargs)
    elif is_beam_gen_mode:
        raise NotImplementedError
    else:
        raise ValueError("Unsupported generation mode")

from typing import List, Union

from transformers.generation.utils import (GenerationMixin,
                                           SampleDecoderOnlyOutput,
                                           SampleEncoderDecoderOutput,
                                           SampleOutput, StoppingCriteriaList,
                                           validate_stopping_criteria)


## copy from GenerationMixin   
def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.sample`] directly. Use generate() instead.
        For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.SampleDecoderOnlyOutput`], [`~generation.SampleEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.SampleEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     TopKLogitsWarper,
        ...     TemperatureLogitsWarper,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id
        >>> model.generation_config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = LogitsProcessorList(
        ...     [
        ...         TopKLogitsWarper(50),
        ...         TemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        >>> outputs = model.sample(
        ...     input_ids,
        ...     logits_processor=logits_processor,
        ...     logits_warper=logits_warper,
        ...     stopping_criteria=stopping_criteria,
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today is a beautiful day, and we must do everything possible to make it a day of celebration.']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            logging_rank_0(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                "debug",
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )
        strict_max_length = model_kwargs.get("strict_max_length", None)
        if return_dict_in_generate:
            return_dict_in_generate = False
            logging_rank_0(
                " Can't enable 'return_dict_in_generate'", "debug"
            )
        if synced_gpus:
            logging_rank_0(
                "Enable 'synced_gpus'", "debug"
            )
        

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        # unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        def convert_left_padding_to_right(input_ids, pad_token_id, eos_token_id):
            if torch.any(input_ids[:,0] == pad_token_id): # has left padding
                rp_ids = torch.ones_like(input_ids)*eos_token_id
                lidx = torch.min(torch.where(input_ids != pad_token_id, torch.arange(input_ids.size(1)).to(input_ids.device), 10000000), dim=1)[0]
                for row in range(input_ids.shape[0]):
                    rp_ids[row, :input_ids.shape[1]-lidx[row]] = input_ids[row, lidx[row]:]
                input_ids = rp_ids

            return input_ids
            
        input_ids = convert_left_padding_to_right(input_ids, pad_token_id, eos_token_id) 
        context_length = torch.max(torch.where(input_ids != pad_token_id, torch.arange(1, input_ids.size(1)+1).to(input_ids.device), -1), dim=1)[0] ## 对于右padding的情况，找出每条样本的真实长度
        start_length = torch.min(context_length)
        batch_size = input_ids.shape[0]
        logging_rank_0(f"Rank-{input_ids.device}: input-shape-{input_ids.shape}", "debug")
        
        # record order
        origin_order = torch.tensor(range(batch_size), device=input_ids.device, dtype=torch.int64)  # 原始输入的编号，按照索引
        cur_order = torch.tensor([], device=input_ids.device, dtype=torch.int64)                    # 正在生成的编号
        cur_tokens = torch.tensor([], device=input_ids.device, dtype=input_ids.dtype)               # 正在生成的token
        output_order = []                                                                           # 结束生成的编号
        output_tokens_lists = []                                                                    # 结束生成的token
        
        # wait_gen_idx = torch.argwhere(context_length > start_length).view(-1)
        # input_ids = input_ids[wait_gen_idx]
        # origin_order = origin_order[wait_gen_idx]
        

        # finished = torch.zeros(input_ids.shape[0], dtype=torch.bool).to(input_ids.device)
        # def switch(next_tokens, col_idx, fill_value=eos_token_id):
        #     is_update = (col_idx >= context_length)
        #     is_eos = (next_tokens == eos_token_id)
        #     finished[is_update&is_eos] = True
        #     next_tokens[finished] = eos_token_id
        #     if col_idx >= input_ids.shape[1]:
        #         return next_tokens
            
        #     is_update = is_update.type_as(next_tokens)
        #     init = input_ids[:,col_idx]
        #     return is_update*next_tokens + (1-is_update)*init


        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        counter = 0
        num_tokens = []
        total_time = 0
        timer = {
            "sync": 0,
            "forward": 0,
            "befo_forward": 0,
            "after_forward": 0
        }
        while True:
            t0 = time.time()
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            t1 = time.time()    
            index = start_length + counter
            
            # init gen batch
            start_gen_idx = torch.argwhere(context_length == index).view(-1)
            
            if start_gen_idx.shape[0] > 0:  # add new sequence
                new_tokens = input_ids[start_gen_idx, :index]
                new_order = origin_order[start_gen_idx]
                cur_tokens = torch.cat((cur_tokens, new_tokens))
                cur_order = torch.cat((cur_order, new_order))
                model_kwargs["past_key_values"] = None
            
            # skip
            if cur_tokens.shape[0] == 0:
                cur_tokens = torch.tensor([], device=input_ids.device, dtype=input_ids.dtype)
                counter += 1
                # print(f"Rank-{input_ids.device}-idx-{index}: Continue") 
                continue
            # use default attention mask
            if "attention_mask" in model_kwargs.keys():
                # logging_rank_0(
                #     "Reset to default attention mask", "debug"
                # )
                model_kwargs["attention_mask"] = torch.ones_like(cur_tokens)
            
            ### 统计推理的计算量
            if model_kwargs.get("past_key_values", None) is None:
                num_tokens.append(list(cur_tokens.shape) + [False])
            else:
                num_tokens.append([cur_tokens.shape[0], cur_tokens.shape[1], True]) # 使用kv cache
            ###
            
            model_inputs = self.prepare_inputs_for_generation(cur_tokens, **model_kwargs)
            # forward pass to get next token
            torch.cuda.synchronize()
            t2 = time.time()
            outputs = self(
                **model_inputs,
                return_dict=True,
                use_cache=model_kwargs["use_cache"] if "use_cache" in model_kwargs.keys() else False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            torch.cuda.synchronize()
            t3 = time.time()
            total_time += t3 - t2

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(cur_tokens, next_token_logits)
            next_token_scores = logits_warper(cur_tokens, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            # next_tokens = switch(next_tokens, index)
            
            # update generated ids, model inputs, and length for next step
            cur_tokens = torch.cat([cur_tokens, next_tokens[:, None]], dim=-1)
            
            # detach stop sequence in this turn
            stop_idx = next_tokens == eos_token_id              # 需要移出batch的样本在cur_order中的idx
            if torch.any(stop_idx):
                finished_sequences = cur_tokens[stop_idx]
                output_tokens_lists.extend(finished_sequences.detach().cpu().tolist())
                output_order.extend(origin_order[cur_order[stop_idx]].tolist())
            
            # process others
            continue_idx = (next_tokens != eos_token_id)        # 需要继续生成的样本在cur_order中的idx
            cur_order = cur_order[continue_idx]
            cur_tokens = cur_tokens[continue_idx]
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if "past_key_values" in model_kwargs.keys():
                new_past_key_values = ()
                for mem_idx in range(len(model_kwargs["past_key_values"])):
                    continue_past_key_values = model_kwargs["past_key_values"][mem_idx][:,:,continue_idx,:,:]
                    new_past_key_values = new_past_key_values + (continue_past_key_values, )
                model_kwargs["past_key_values"] = new_past_key_values

            if streamer is not None:
                streamer.put(next_tokens.cpu())
            
            # stop when each sentence is finished
            if batch_size == len(output_order):
                logging_rank_0(f"Rank-{input_ids.device}-idx-{index}: Break-len", "debug") 
                this_peer_finished = True
            
            if strict_max_length and cur_tokens.shape[1] >= strict_max_length:
                logging_rank_0(f"Rank-{input_ids.device}-idx-{index}: Break-max-len", "debug") 
                this_peer_finished = True
                
            # stop if we exceed the maximum length
            if stopping_criteria(cur_tokens, scores):
                logging_rank_0(f"Rank-{input_ids.device}-idx-{index}: Break-criteria", "debug") 
                this_peer_finished = True
                
            t4 = time.time()
            
            timer["sync"] += t1 - t0
            timer["befo_forward"] += t2 - t1
            timer["forward"] += t3 - t2
            timer["after_forward"] += t4 - t3

            if this_peer_finished and not synced_gpus:
                break
            
            counter += 1
          
        if streamer is not None:
            streamer.end()
        
        if batch_size != len(output_order): # 存在样本未生成完，或未开始生成
            if cur_tokens.shape[0] > 0:
                output_tokens_lists.extend(cur_tokens.detach().cpu().tolist())
                output_order.extend(cur_order.tolist())
            else:
                logging_rank_0("input_ids longer than 'max_length'", "debug")
        
        output_tokens_lists = [tokens for _, tokens in sorted(zip(output_order, output_tokens_lists))]
        output_sequences = pad_sequence(
            [torch.tensor(item, dtype=input_ids.dtype, device=input_ids.device) for item in output_tokens_lists], batch_first=True, padding_value=pad_token_id
        )
        
        logging_rank_0(f"Rank-{input_ids.device}-idx-{index}: Forward-total-{timer}", "debug")

        # if return_dict_in_generate:
        #     if self.config.is_encoder_decoder:
        #         return SampleEncoderDecoderOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             encoder_attentions=encoder_attentions,
        #             encoder_hidden_states=encoder_hidden_states,
        #             decoder_attentions=decoder_attentions,
        #             cross_attentions=cross_attentions,
        #             decoder_hidden_states=decoder_hidden_states,
        #         )
        #     else:
        #         return SampleDecoderOnlyOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             attentions=decoder_attentions,
        #             hidden_states=decoder_hidden_states,
        #         )
        # else:
        torch.cuda.empty_cache()
        gc.collect()
        return output_sequences, {"total_tokens": num_tokens, "total_time": total_time}

#****************************************
# .             WARNING                 #
## rewrite GenerationMixin.sample      ##
GenerationMixin.sample = _sample
