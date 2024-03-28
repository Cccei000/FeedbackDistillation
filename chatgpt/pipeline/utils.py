# encoding=utf-8
import json
from collections import defaultdict
from typing import Dict, Optional

import datasets as ds
import torch.distributed as dist
from tqdm import tqdm
from transformers import LlamaTokenizer
from chatgpt.utils import is_rank_0, logging_rank_0


LOGGED_WARNING = {
    "prefix": False,
    "seperator": False,
}

def check_data_fromat(idx:int, item:dict, prefix:Optional[Dict[str,str]]=None, seperator:Optional[str]=None) -> bool:
    """检查数据集样本的格式

    Args:
        idx (int): 样本ID
        item (dict): 单个数据样本
        prefix (Optional[Dict[str,str]], optional): 对话角色的prompt. Defaults to None.
        seperator (Optional[str], optional): 不同角色之间的分隔符. Defaults to None.

    Returns:
        bool: _description_
    """
    
    result = True
    
    global LOGGED_WARNING
    
    ### 检查 query
    if item.get("query", None) is None:
        logging_rank_0(f"Detected incomplete data. Missing key 'query' on line {idx}.", "warning")
        return False  
    elif isinstance(item["query"], list):
        if prefix is None:
            logging_rank_0(f"Detected Multi-turn query. 'prefix' is required.", "error")
            return False
        for q in item["query"]:
            role = q.get("role", None)
            text = q.get("text", None)
            if role is None:
                logging_rank_0(f"Detected incomplete data. Missing key 'role' on line {idx}.", "warning")
                result = False
            elif role not in prefix.keys():
                logging_rank_0(f"Detected illegal 'role' on line {idx}, should in {set(prefix.keys())}", "warning")
                result = False
            if text is None:
                logging_rank_0(f"Detected incomplete data. Missing key 'text' on line {idx}.", "warning")
                result = False
    elif isinstance(item["query"], str):
        if prefix is not None and not LOGGED_WARNING["prefix"]:
            logging_rank_0(f"Detected string-type query. 'prefix' won't be used.", "warning")
            LOGGED_WARNING["prefix"] = True
        if seperator is not None and not LOGGED_WARNING["seperator"]:
            logging_rank_0(f"Detected string-type query. 'multiturn_seperator' won't be used.", "warning")
            LOGGED_WARNING["seperator"] = True
    
    ### 检查 task
    task = item.get("task", None)
    if task is not None and not isinstance(task, str):
        logging_rank_0(f"Detected wrong data type on line {idx}. Key 'task' should be string-type.", "warning")
        result = False
    
    ### 检查 golden_res
    golden_res = item.get("golden_res", None)
    if golden_res is not None and not isinstance(golden_res, str):
        logging_rank_0(f"Detected wrong data type on line {idx}. Key 'golden_res' should be string-type.", "warning")
        result = False

    ### 检查 preference
    preference = item.get("preference", None)
    if preference is not None:
        if not isinstance(preference, list):
            logging_rank_0(f"Detected wrong data type on line {idx}. Key 'preference' should be a list of strings.", "warning")
            result = False
        else:
            for res in preference:
                if not isinstance(res, str):
                    logging_rank_0(f"Detected wrong data type on line {idx}. Key 'preference' should be a list of strings.", "warning")
                    result = False
                    break
    
    return result


def load_jsonline_data(path:str, prefix:Optional[Dict[str,str]]=None, seperator:Optional[str]=None, disable_bar:bool=False) -> ds.Dataset:
    """读取jsonline数据

    Args:
        path (str): _description_

    Returns:
        ds.Dataset: _description_
    """
    
    ds.disable_caching()
    
    items = []
    with open(path, "r") as f:
        for line in f.readlines():
            items.append(json.loads(line))
    
    data_dict = defaultdict(list)
    
    for idx, item in tqdm(enumerate((items)), disable=disable_bar):
        
        if not check_data_fromat(idx, item, prefix, seperator):
            continue
        
        data_dict["query"].append(item["query"])
        data_dict["preference"].append(item.get("preference", []))
        data_dict["golden_res"].append(item.get("golden_res", ""))
        data_dict["task"].append(item.get("task", "default"))
        data_dict["pairs"].append(item.get("pairs", None))
        
    dataset = ds.Dataset.from_dict(data_dict)
    
    return dataset


def concat_prompt(dataset:ds.Dataset, prefix:Optional[Dict[str,str]]=None, seperator:Optional[str]=None) -> ds.Dataset:
    """使用Prompt拼接对话信息

    Args:
        dataset (ds.Dataset): 未拼接的对话数据集
        prefix (Optional[Dict[str,str]], optional): 对话角色的prompt. Defaults to None.
        seperator (Optional[str], optional): _description_. Defaults to None.

    Returns:
        ds.Dataset: 拼接后的对话数据集
    """
    
    def _concat(item):
        query = item["query"]
        if isinstance(query, str):
            return { "query": query }
        query_list = []
        for i,q in enumerate(query):
            prefix_i = prefix[q['role']].replace("{round_number}", str(int(i/2+1)))
            query_list.append(f"{prefix_i}{q['text']}")
        
        query_list.append(prefix["model"].replace("{round_number}", str(int(i/2+1))))
        
        return {
            "query": seperator.join(query_list) if seperator is not None else "".join(query_list)
        }
        
    return dataset.map(function=_concat, batched=False)


def save_dataset_to_jsonl(dataset:ds.Dataset, path:str):
    
    if dist.is_initialized() and is_rank_0():
        jsonl_str = []
        for item in dataset:
            jsonl_str.append(json.dumps(item, ensure_ascii=False))
        jsonl_str = "\n".join(jsonl_str)
        
        try:
            with open(path, "w+") as f:
                f.write(jsonl_str)
        except Exception:
            logging_rank_0(f"Fail to save splited rm data to '{path}'", "warning")
        else:
            logging_rank_0(f"Save splited rm data to '{path}'")
    
    # if dist.is_initialized():
    #     dist.barrier()
        
    return
