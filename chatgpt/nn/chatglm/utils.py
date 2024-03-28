# encoding=utf-8
import torch
from chatgpt.nn.chatglm import ChatGLMConfig, ChatGLMForConditionalGeneration, ChatGLMModel


def load_chatglm_causal_lm_ckpt(ckpt_path:str) -> ChatGLMForConditionalGeneration:
    print(f"Load ChatGLMConditionalGeneration Model from '{ckpt_path}'...", end="")
    config = ChatGLMConfig.from_pretrained(ckpt_path)
    model = ChatGLMForConditionalGeneration.from_pretrained(ckpt_path, config=config)
    print("Finished!")
    return model


def load_chatglm_lm_ckpt(ckpt_path:str) -> ChatGLMModel:
    print(f"Load ChatGLM Model from '{ckpt_path}'...", end="")
    config = ChatGLMConfig.from_pretrained(ckpt_path)
    model = ChatGLMModel.from_pretrained(ckpt_path, config=config)
    print("Finished!")
    return model


def get_masks(input_ids:torch.Tensor, bos_token_id:int, pad_token_id:int, device:torch.device) -> torch.Tensor:
    batch_size, seq_length = input_ids.shape
    bos_locs = torch.where(input_ids == bos_token_id)
    
    attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
    attention_mask.tril_()
    for i, bos_idx in zip(bos_locs[0], bos_locs[1]):
        attention_mask[i, :bos_idx, :bos_idx] = 1
        
    pad_mask = input_ids != pad_token_id
    attention_mask = attention_mask * pad_mask[:, None, :] * pad_mask[:, :, None]
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()

    return attention_mask


def get_position_ids(input_ids:torch.Tensor, gmask_token_id:int, bos_token_id:int, pad_token_ids:int, device:torch.device, gmask=False, position_encoding_2d=True) -> torch.Tensor:
    batch_size, seq_length = input_ids.shape

    mask_positions = [seq.tolist().index(gmask_token_id) for seq in input_ids]
    context_lengths = [seq.tolist().index(bos_token_id) for seq in input_ids]
    if position_encoding_2d:
        # position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        position_ids = (input_ids != pad_token_ids).cumsum(dim=-1) # 忽略左padding
        for i, context_length in enumerate(context_lengths):
            position_ids[i, context_length:] = mask_positions[i]
        block_position_ids = [torch.cat((
            torch.zeros(context_length, dtype=torch.long),
            torch.arange(seq_length - context_length, dtype=torch.long) + 1
        )) for context_length in context_lengths]
        block_position_ids = torch.stack(block_position_ids, dim=0).to(device=device)
        position_ids = torch.stack((position_ids, block_position_ids), dim=1)
    else:
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        if not gmask:
            for i, context_length in enumerate(context_lengths):
                position_ids[context_length:] = mask_positions[i]

    return position_ids.to(device=device)
