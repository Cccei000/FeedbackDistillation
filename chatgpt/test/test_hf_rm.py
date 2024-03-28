
# from chatgpt.dataset import RMCollator
# from chatgpt.nn.llama import LlamaHFRewardModel
# from chatgpt.pipeline.utils import set_tokenizer
"""✨✨✨为了让hf测试脚本不安装chatgpt项目也能够运行，将上面的import代码改为复制到这里✨✨✨"""

from typing import Callable, Optional
import pandas as pd
from datasets import Dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, LlamaTokenizer, LlamaConfig, LlamaModel

class RMCollator():
    """
    Collator for reward model

    Args:
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(self, tokenizer: Callable, max_length: int, query_key:str="query", response_key:str="responses") -> None:
        self.tokenizer = tokenizer 
        self.max_seq_length = max_length
        self.query_key = query_key
        self.response_key = response_key

    def get_inputs(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt",padding=True,truncation=True,max_length=self.max_seq_length
        )
        return inputs.input_ids, inputs.attention_mask

    def __call__(self, batch):
        pair_length = [0]
        input_ids = []
        action_masks = []
        input_text = []
        l = 0
        eos_token = self.tokenizer.eos_token # "</s>"
        for d in batch:
            query = d[self.query_key]
            query_ids = self.tokenizer(query, max_length=self.max_seq_length-1, truncation=True)["input_ids"]
            for i, r in enumerate(d[self.response_key]):
                if isinstance(r, str):
                    r_text = r
                else:
                    r_text = r['text'] if r['text'] is not None else ""
                text = query + r_text
                response = self.tokenizer(r_text+eos_token)["input_ids"][1:]
                input_ids.append(torch.Tensor(query_ids+response))
                action_masks.append(torch.Tensor([0]*len(query_ids) + [1]*len(response)))
                input_text.append(text)
                l+=1
            pair_length.append(l)
        
        input_ids = [ids[:self.max_seq_length] for ids in input_ids]
        action_masks = [m[:self.max_seq_length] for m in action_masks]
        input_ids = zero_pad_sequences(input_ids, side= 'right', padding_value=self.tokenizer.pad_token_id).long()
        action_masks = zero_pad_sequences(action_masks, side= 'right', padding_value=0).long()
        attn_mask = input_ids.not_equal(self.tokenizer.pad_token_id).long()
        if self.tokenizer.eos_token_id is not None and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            last_index =  torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in attn_mask], dtype=torch.int64)
            for i,ind in enumerate(last_index):
                if ind+1<self.max_seq_length:
                    attn_mask[i,ind+1]=1
        output = {
            "input_ids":input_ids, 
            "attention_mask":attn_mask,
            "action_mask":action_masks,
            "pair_length":torch.Tensor(pair_length).long(),
            "input_text":input_text
        }
        return output
    
class LlamaHFRewardModel(PreTrainedModel):
    # hf model for reward model
    # load like: model = LlamaHFRewardModel.from_pretrained(hf_rm_paht,granularity="sample").to(torch.bfloat16).cuda().eval()
    # granularity must be in ["sample","token","token_sample_mix"]

    config_class =LlamaConfig
    
    def __init__(self, config, granularity="sample"):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.value_head = torch.nn.Linear(config.hidden_size, 1) 
        self.token_value_head = torch.nn.Linear(config.hidden_size, 1)
        assert granularity in ["sample","token","token_sample_mix"], f"RM granularity must be in [\"sample\",\"token\",\"token_sample_mix\"], current RM granularity: {granularity}"
        self.granularity = granularity
    
    def forward(self,
                input_ids: torch.LongTensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(input_ids,attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        if self.granularity == "token":
            values = self.value_head(hidden_states).squeeze(-1)
            return values
        
        if attention_mask is None:
            last_hidden_states = hidden_states[:, -1]
        else:
            last_index =  torch.tensor([(a == 1).nonzero(as_tuple=True)[0][-1].item() for a in attention_mask], dtype=torch.int64)
            last_hidden_states = hidden_states[torch.arange(hidden_states.shape[0]), last_index]
        if self.granularity == "sample":
            values = self.value_head(last_hidden_states).squeeze(-1)
            return values
        if self.granularity == "token_sample_mix":
            token_values = self.token_value_head(hidden_states).squeeze(-1) #  (bs,len_seq)
            values = self.value_head(last_hidden_states).squeeze(-1) # (bs,)
            token_values.index_put_(indices=[torch.arange(token_values.shape[0]), last_index],values=values)
            return token_values
        
def zero_pad_sequences(sequences, side: str = 'left', padding_value: int = 0) -> torch.Tensor:
    assert side in ('left', 'right')
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == 'left' else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=padding_value))
    return torch.stack(padded_sequences, dim=0)

def set_tokenizer(tokenizer_path, model_type):
    from tokenizers import AddedToken
    if model_type == "llama_13B":
        human = AddedToken("<human>",lstrip=False,rstrip=False,single_word=False,normalized=True)
        bot = AddedToken("<bot>",lstrip=False,rstrip=False,single_word=False,normalized=True)
        llama_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        llama_tokenizer.add_special_tokens({"additional_special_tokens":[human, bot]})
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
    elif model_type == "llama2_13B":
        # logging_rank_0(f"tokenizer type: {model_type},tokenizer path: {tokenizer_path}")
        special_token_dict = {'pad_token': '</s>'}
        llama_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        llama_tokenizer.add_special_tokens(special_token_dict)
    else:
        llama_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        # llama_tokenizer.pad_token_id = 3
    return llama_tokenizer

def test():
    data_path = "/cognitive_comp/liangyuxin/workspace/pipeline/RM_LLAMA2_13B_0829/0829/rm_data/test.jsonl"
    tokenizer_path = "/cognitive_comp/songzhuoyang/models/llama2_13B_sft/step_26k_hf/"
    model_path = "/cognitive_comp/liangyuxin/workspace/pipeline/RM_LLAMA2_13B_0829/0829/ckpt/reward_model/global_step14005_hf"
    granularity = "sample"
    
    tokenizer = set_tokenizer(tokenizer_path, "llama2_13B")
    max_length = 2048
    df = pd.read_json(data_path, lines=True)
    dataset = Dataset.from_pandas(df)
    collor = RMCollator(tokenizer, max_length, query_key="query", response_key="preference")
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collor,shuffle=False)
    model = LlamaHFRewardModel.from_pretrained(model_path,granularity=granularity).to(torch.bfloat16).cuda().eval()
    all_rewards = []
    with torch.no_grad():
        for d in dataloader:
            outputs = model(input_ids=d["input_ids"].to(model.device),attention_mask=d["attention_mask"].to(model.device))
            if granularity !="sample":
                outputs = outputs* d["action_mask"].to(model.device) # only keep the reward of the response
            all_rewards.append(outputs.cpu().tolist())
    df["reward"] = all_rewards
    save_path = data_path.replace(".jsonl","rm_scored.jsonl")
    df.to_json(save_path, orient="records", lines=True,force_ascii=False)
    print("save to ",save_path)
    return

if __name__ == "__main__":
    test()