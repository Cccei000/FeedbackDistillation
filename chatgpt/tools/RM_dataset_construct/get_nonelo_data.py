import os
import re
import random
import jsonlines
import pandas as pd
import json
import argparse
from datasets import load_from_disk,concatenate_datasets,DatasetDict,Dataset
from methods import (
    punc_repetition,
    text_repetition,
    delete_punc,
    void_text,
    shuffle_query_response,
    change_number,
    split_dataset,
)

def set_args(parser):
    parser.add_argument("--save_base_path", type=str, default=None)
    parser.add_argument("--data_config", type=str, default=None)
    return parser

def read_path_list(path_list,config,name):
    dfs = []
    for i,d in enumerate(path_list):
        path = d['path']
        df = pd.read_json(path,lines=True)
        if d['number']!="all":
            num = min(d['number'],df.shape[0])
            df = df.sample(n=num).reset_index(drop=True)
        dfs.append(df)
        config[name][i]['number'] = df.shape[0]
    df = pd.concat(dfs).reset_index(drop=True)
    return df,config

def single_query_add_role(query_list):
    query_dict = []
    for q in query_list:
        query_dict.append([{
            "role":"human",
            "text":q
        }])
    return query_dict

def responses_add_role(responses_list):
    responses_dict = []
    for responses in responses_list:
        r_dict = []
        for r in responses:
            r_dict.append(
                {
                    "role":"model",
                    "text":r
                }
            )
        responses_dict.append(r_dict)
    return responses_dict

def multi_query_add_role(query_list):
    # 英文数据集中多轮对话
    query_dict = []
    for queries in query_list:
        q_dict = []
        for i,q in enumerate(queries):
            role = "human" if i%2==0 else "model"
            q_dict.append({"role":role,"text":q})
        query_dict.append(q_dict)
    return query_dict

def add_role(d):
    if "query" not in d or "responses" not in d:
        raise Exception("Missing required key: query/responses")
    query_list = d['query']
    responses_list = d["responses"]
    if isinstance(query_list[0], str):
        query_dict = single_query_add_role(query_list)
    elif isinstance(query_list[0], list):
        query_dict = multi_query_add_role(query_list)
    else:
        raise Exception("query not str or list, what is it?")
    
    if isinstance(responses_list[0],list) and isinstance(responses_list[0][0],str):
        responses_dict = responses_add_role(responses_list)
    else:
        raise Exception("responses not list of str, what is it?")
    if "timestamp" in d:
        timestamp = d["timestamp"]
    else:
        timestamp = [""]*len(query_dict)
    d_new = {"query":query_dict,"responses":responses_dict,"timestamp":timestamp}
    return d_new

def dialogue_add_role(d):
    # 标注数据集中多轮对话
    user_texts = d['user_texts']
    bot_texts = d["bot_texts"]
    query_dict = []
    for user_text,bot_text in zip(user_texts,bot_texts):
        query = []
        query.append({"role":"human","text":user_text[0]})
        for user_t,bot_t in zip(user_text[1:],bot_text):
            query.append({"role":"model","text":bot_t})
            query.append({"role":"human","text":user_t})
        query_dict.append(query)
    responses_list = d["responses"]
    responses_dict=responses_add_role(responses_list)
    if "timestamp" in d:
        timestamp = d["timestamp"]
    else:
        timestamp = [""]*len(query_dict)
    d_new = {"query":query_dict,"responses":responses_dict,"timestamp":timestamp}
    return d_new


def response_clip(d):
    query_list = d['query']
    responses_list = [r[:2] for r in d["responses"]]
    d_new = {"query":query_list,"responses":responses_list}
    return d_new

# 1. 从标注系统收集数据
def gather_data_from_labeler(save_path,config,name):
    # usable_dataname = config[name][0]['dataset_name']
    base_path = config[name][0]['path']
    excluded_datasets = ["bkp_0515","bkp_0519","test_ldy_dataset","dev_dialogue_rw_dataset","dev_rw_dataset","test_rw_dataset","dialogue_rw_dataset"]
    excluded_keys = ['meta','candidate','noname',"liuxingyue","yangzijuan","zhaoliang","dislike"]
    n=50
    ds_all= []
    # ds_all_dialogue = []
    data_nums = {}
    for dataname in os.listdir(base_path):
        if dataname not in excluded_datasets and "bkp" not in dataname:# and "math" in dataname:
            print(dataname)
            data_num = 0
            ds = load_from_disk(base_path+dataname)
            ds_tmp = []
            for key in ds:
                if key not in excluded_keys and len(ds[key])>n:
                    # if key in ['qa_rw_dataset','online_rw_dataset','writing_0302_rw_dataset']:
                    data_num+=(len(ds[key])-n)
                    ds_tmp.append(ds[key].select(range(n,len(ds[key]))))
                    # else:
                    #     data_num+=len(ds[key])
                    #     ds_tmp.append(ds[key])
            if len(ds_tmp)>=1:
                ds_tmp = concatenate_datasets(ds_tmp)
                if len(ds_tmp)>0:
                    d = ds_tmp.to_dict()
                    d = dialogue_add_role(d) if 'user_texts' in d else add_role(d)
                    d['source'] = [dataname]*len(d['query'])
                    ds = Dataset.from_dict(d)
                    ds_all.append(ds)
                    print(data_num)
                    data_nums[dataname]=data_num
    ds = concatenate_datasets(ds_all)
    # ds = Dataset.from_dict(d)
    print(ds)
    ds_save = split_dataset(ds)
    ds_save.save_to_disk(save_path)
    config['labeled'][0]["number"] = len(ds)
    config['labeled'][0]["detail"] = data_nums
    print(ds_save)
    return config


# 2. 构造对抗数据
def construct_adversarial_dataset(save_path,config,name):
    path = config[name][0]['path']
    adversarial_sample_num = config[name][0]['number']
    methods_dict = {
        "punc_repetition":punc_repetition,
        "text_repetition":text_repetition,
        "delete_punc":delete_punc,
        "void_text":void_text,
        }
    df = pd.read_json(path,lines=True)
    pair_list = [{"query":q,"responses":[r]} for q,r in zip(df['query'],df['responses'])]
    new_pair_list = []
    for key,method in methods_dict.items():
        for i in range(adversarial_sample_num):
            pair = random.choice(pair_list)
            response = method(pair["responses"][0])
            new_pair_list.append({"query":pair['query'],"responses":[pair["responses"][0],response]})
        config[name][0][key] = adversarial_sample_num
    
    pair = random.sample(pair_list,adversarial_sample_num)
    query_list = [d['query'] for d in pair]
    response_list = [d['responses'][0] for d in pair]
    new_response_list=random.sample(response_list,len(response_list))
    for q,r,new_r in zip(query_list,response_list,new_response_list):
        new_pair_list.append({"query":q,"responses":[r,new_r]})
    config[name][0]["random_shuffle"] = adversarial_sample_num
    d = {"query":[d["query"] for d in new_pair_list],"responses":[d["responses"] for d in new_pair_list]}
    d = add_role(d)
    d['source'] = [name]*len(d['query'])
    ds = Dataset.from_dict(d)
    ds_save = split_dataset(ds)
    ds_save.save_to_disk(save_path)

    return config

def construct_adversarial_dataset_math(save_path,config,name):
    path = config[name][0]['path']
    adversarial_sample_num = config[name][0]['number']
    df = pd.read_json(path,lines=True)
    pair_list = [{"query":q,"responses":[r]} for q,r in zip(df['query'],df['responses'])]
    new_pair_list = []
    for i in range(adversarial_sample_num):
        pair = random.choice(pair_list)
        response = change_number(pair["responses"][0])
        if response is not None:
            new_pair_list.append({"query":pair['query'],"responses":[pair["responses"][0],response]})
    config[name][0]["number"] = len(new_pair_list)
    d = {"query":[d["query"] for d in new_pair_list],"responses":[d["responses"] for d in new_pair_list]}
    d = add_role(d)
    d['source'] = [name]*len(d['query'])
    ds = Dataset.from_dict(d)
    ds_save = split_dataset(ds)
    ds_save.save_to_disk(save_path)
    return config

# 3.good vs mindbot generate data pair
def pair_data_inter_models(save_path,config,name):
    # good data from: 1.gpt generated 2.labeled data
    paths_list = config[name]
    df,config = read_path_list(paths_list,config,name)
    d = add_role(df.to_dict(orient="list"))
    d['source']=[name]*len(d['query'])
    ds = Dataset.from_dict(d)
    ds_save = split_dataset(ds)
    ds_save.save_to_disk(save_path)
    return config

def ai_feedback(save_path,config,name):
    path_list = config[name]
    df,config = read_path_list(path_list,config,name)
    df = df[["gpt_response"]]
    prompts = [responses[0].split("\n\nAssistant:")[0].split("Human:")[1] for responses in df["gpt_response"].tolist()]
    responses = [[responses[i].split("\n\nAssistant:")[1] for i in range(len(responses))] for responses in df["gpt_response"].tolist()]
    # pair_list = [{"query":q,"responses":r} for q,r in zip(prompts,responses)]
    pair_dict = {"query":prompts,"responses":responses}
    d = add_role(pair_dict)
    ds = Dataset.from_dict(d)
    ds_save = split_dataset(ds)
    ds_save.save_to_disk(save_path)

    return config

def other_dataset(save_path,config,name):
    path_list = config[name]
    df,config = read_path_list(path_list,config,name)
    d = add_role(df.to_dict(orient="list"))
    d['source']=[name]*len(d['query'])
    ds = Dataset.from_dict(d)
    ds_save = split_dataset(ds)
    ds_save.save_to_disk(save_path)
    return config

def get_nonelo_data(data_config,save_base_path):
    with open(data_config, 'r') as f:
        config = json.load(f)
        print(config)
    
    dataset_dict = {
        "labeled":gather_data_from_labeler,
        "adversarial":construct_adversarial_dataset,
        "construct_compare":pair_data_inter_models,
        "ai_feedback":ai_feedback,
        "other_dataset":other_dataset,
        "multiturn":other_dataset,
        "math_adversarial":construct_adversarial_dataset_math,
        }
    
    save_dataset_paths = []
    for name in config:
        print(name)
        method = dataset_dict[name]
        print("processing",name)
        save_path = os.path.join(save_base_path,name)
        config = method(save_path,config,name=name)
        save_dataset_paths.append(save_path)
    ds_train = []
    ds_dev = []
    ds_test = []
    for path in save_dataset_paths:
        print(path)
        ds = load_from_disk(path)
        print("before filter",ds)
        ds = ds.filter(lambda x: len(x["responses"])>1 and not any([r["text"] is None for r in x["responses"]]),load_from_cache_file=False)
        print("after filter",ds)
        ds_train.append(ds['train'])
        ds_dev.append(ds['dev'])
        ds_test.append(ds['test'])
    ds = DatasetDict()
    ds['train']=concatenate_datasets(ds_train).shuffle(seed=42)
    ds['dev']=concatenate_datasets(ds_dev)
    ds['test']=concatenate_datasets(ds_test)
    
    return ds,config

def main(args):
    ds,config = get_nonelo_data()
    ds.save_to_disk(args.save_base_path+"/merged_dataset/")
    print(config)
    with open(args.save_base_path+'/merged_dataset/dataset_config.json', 'w') as json_file:
        json.dump(config, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = set_args(parser)
    args = parser.parse_args()
    main(args)
