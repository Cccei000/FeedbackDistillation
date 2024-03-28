import os
import datasets
import json
import datasets
import json
import os
import pandas as pd
import argparse
from openskill import Rating, predict_rank
from methods import (
    split_dataset,
)

datasets.disable_caching()

def set_args(parser):
    parser.add_argument("--save_base_path", type=str, default=None)
    # parser.add_argument("--elo_ds_path", type=str, default=None)
    parser.add_argument("--back_up_ds", type=str, default=None)

    return parser


def responses_add_role(responses_list):
    responses_dict = []
    for responses in responses_list:
        responses_dict.append(
            {
                "role":"model",
                "text":responses
            }
        )
    return responses_dict

def add_role(querys,responses,response_compares):
    user_texts = querys
    bot_texts = responses
    query_dict = []
    query_dict.append({"role":"human","text":user_texts[0]})
    for user_text,bot_text in zip(user_texts[1:],bot_texts):
        query_dict.append({"role":"model","text":bot_text})
        query_dict.append({"role":"human","text":user_text})
    responses_list = response_compares
    responses_dict=responses_add_role(responses_list)
    d = {"query":query_dict,"responses":responses_dict}
    return d

def get_elo_data(base_path):
    dirs = os.listdir(base_path)
    fix_file = os.listdir("/cognitive_comp/hetingting/data/errordata/summary_uptodate")
    all_ds = []
    config={}
    for dir in dirs:
        if "dev" not in dir:
            print(dir)
            save_data = []
            meta = datasets.load_from_disk(base_path+dir+"/meta")
            candidate = datasets.load_from_disk(base_path+dir+"/candidate")
            id2query = {
                d["id"]: (d["query"],d["response"]) for d in candidate
            }
            fix_set = set()
            if dir+".json" in fix_file:
                fix = json.load(open("/cognitive_comp/hetingting/data/errordata/summary_uptodate/"+dir+".json"))
                for k in fix:
                    res = [f[1] for f in fix[k]]
                    res = [item for sublist in res for item in sublist]
                    for r in res:
                        fix_set.add(r)
            def process(x):
                responses = json.loads(x["responses"])
                mu = json.loads(x["mu"])
                sigma = json.loads(x["sigma"])
                timestamp = json.loads(x["timestamp"])
                for r,m,s in zip(responses,mu,sigma):
                    if r in fix_set:
                        responses.remove(r)
                        mu.remove(m)
                        sigma.remove(s)
                if len(responses)<2:
                    return {
                        "responses": responses
                    }
                rates = []
                for m, s in zip(mu, sigma):
                    rates.append([Rating(mu=m, sigma=s)])
                rank = predict_rank(teams=rates)
                new_responses = [None] * len(rank) # 占位符
                for i, r in enumerate(rank):
                    cur_rank = r[0] - 1
                    # 会出现平分的情况，平分的情况直接往前平移
                    while new_responses[cur_rank] is not None:
                        cur_rank = cur_rank - 1
                    new_responses[cur_rank] = responses[i]
                new_responses = [r for r in new_responses if r is not None]
                return {
                    "responses": new_responses,
                    "timestamp":timestamp[0]
                }
            new_ds = meta.map(process)
            save_data = []
            for d in new_ds:
                query,response = id2query[d["id"]]
                response_compare = d["responses"]
                d_new = add_role(query,response,response_compare)
                d_new["source"] = dir
                d_new["timestamp"] = d["timestamp"]

                save_data.append(d_new)
            save_ds = datasets.Dataset.from_list(save_data)
            config[dir] = len(save_ds)
            all_ds.append(save_ds)
    all_ds = datasets.concatenate_datasets(all_ds)
    
    return all_ds,config

def main(args):
    all_ds,config = get_elo_data()
    ds_save = split_dataset(all_ds)

    if args.back_up_ds:
        non_elo_ds = datasets.load_from_disk(args.back_up_ds)
        config["non_elo"] = args.back_up_ds
        # concate ds_save and dataset
        train = datasets.concatenate_datasets([non_elo_ds["train"],ds_save['train']])
        dev = datasets.concatenate_datasets([non_elo_ds["dev"],ds_save['dev']])
        test = datasets.concatenate_datasets([non_elo_ds["test"],ds_save['test']])
        ds_save = datasets.DatasetDict()
        ds_save['train']=train
        ds_save['dev']=dev
        ds_save['test']=test
        ds_save.save_to_disk(args.save_base_path)
    else:
        ds_save.save_to_disk(args.save_base_path)
    with open(args.save_base_path+'/dataset_config.json', 'w') as json_file:
        json.dump(config, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = set_args(parser)
    args = parser.parse_args()
    main(args)


        

