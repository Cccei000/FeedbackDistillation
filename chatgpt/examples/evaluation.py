import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from typing import List
import torch.nn.functional as F

def zero_pad_sequences(sequences: List[torch.Tensor], side: str = 'left', padding_value: int = 0) -> torch.Tensor:
    assert side in ('left', 'right')
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == 'left' else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=padding_value))
    return torch.stack(padded_sequences, dim=0)

def generate(queries: List[str], tokenizer: AutoTokenizer, model: LlamaForCausalLM, device: int=0, apply_prefix_func=None, **generate_kwargs):
    def _apply_prefix(query):
        return f"<human>:{query.strip()}\n<bot>:"
    
    apply_prefix_func = apply_prefix_func or _apply_prefix

    def _tokenizing(queries):

        input_ids = []
        for query in queries:
            query = apply_prefix_func(query)
            input_ids.append(torch.tensor(tokenizer(query).input_ids))
        inputs = zero_pad_sequences(input_ids, side="left", padding_value=generate_kwargs["pad_token_id"])
        return inputs


    input_ids = _tokenizing(queries).to(device)
    pad_token_id = generate_kwargs["pad_token_id"]
    input_attention_mask = input_ids.not_equal(pad_token_id).to(dtype=torch.bool, device=device)
    sequences = model.generate(
        input_ids.to(0), attention_mask=input_attention_mask, **generate_kwargs)
    output = []
    for i,seq in enumerate(sequences):
        out_text = llama_tokenizer.decode(seq.tolist()[len(input_ids[i]):], skip_special_tokens=False)
        output.append(out_text.replace('<s>','').replace('</s>',''))
    return output


if __name__ == '__main__':

    tk_path = '/cognitive_comp/liuhan/checkpoints/llama-neox-sft/13B-c-pretrain-tokenizer/'
    model_path = '/cognitive_comp/wanghao/models/llama_sft/llama_13B_v2_S2.2_S3_S3.1_rlhf_0601_RM13B0525_step44_hf'

    model = LlamaForCausalLM.from_pretrained(model_path).half().cuda()
    llama_tokenizer = AutoTokenizer.from_pretrained(tk_path) 

    generate_kwargs = {
        "do_sample": True,
        "top_p": 0.85,   
        "top_k": 0,
        "max_length": 2048,
        "repetition_penalty": 1.0,
        "temperature": 0.8,
        "pad_token_id": llama_tokenizer.eos_token_id,
        "eos_token_id": llama_tokenizer.eos_token_id,
    }

    import pandas as pd
    # fname = '/cognitive_comp/wanghao/experiments/test/check模型测评query正确性.xlsx'
    fname = '/cognitive_comp/wanghao/experiments/test/多轮评测数据集.xlsx'

    # queries = pd.read_excel(fname)['query'].tolist()
    queries = pd.read_excel(fname)['多轮对话'].tolist()
    queries = [query.replace('$A$', '<human>').replace('$B$', '<bot>') for query in queries]

    from tqdm import tqdm
    
    responses = []
    bz = 10
    for i in tqdm(range(0,len(queries),bz)):
        qurey_batch = queries[i:i+bz]
        response_batch = generate(qurey_batch, tokenizer=llama_tokenizer, model=model, **generate_kwargs)
        responses.extend(response_batch) 

    with pd.ExcelWriter('./llama_13B_v2_S2.2_S3_S3.1_rlhf_0603_RM13B0525_step54_hf_dialogue.xlsx') as writer:
    # with pd.ExcelWriter('./llama_13b_pool_0512_step169_13BRM_from_s4_1900.xlsx') as writer:
    df = pd.DataFrame({'query':queries, 'response':responses})
    df.to_excel(writer)  