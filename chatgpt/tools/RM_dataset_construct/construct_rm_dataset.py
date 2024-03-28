import argparse
import datasets
import json
from get_nonelo_data import get_nonelo_data
from elo_rank_data import get_elo_data
from methods import split_dataset,debias

def set_args(parser):
    parser.add_argument("--save_base_path", type=str, default=None)
    parser.add_argument("--backup_ds_path", type=str, default=None)
    parser.add_argument("--data_config", type=str, default=None)
    parser.add_argument("--debias", action="store_true")
    parser.add_argument("--save_as_ds", action="store_true")
    return parser

def main(args):
    base_path = "/cognitive_comp/songzhuoyang/dialogue_rm_datasets/"
    elo_ds,config = get_elo_data(base_path)
    print("elo_ds:",elo_ds)

    if args.backup_ds_path:
        non_elo_ds = datasets.load_from_disk(args.backup_ds_path)
        non_elo_ds = datasets.concatenate_datasets([non_elo_ds["train"],non_elo_ds["dev"],non_elo_ds["test"]])
        config["non_elo"] = args.backup_ds_path
    else:
        non_elo_ds,config_nonelo = get_nonelo_data(args.data_config,args.save_base_path)
        non_elo_ds = datasets.concatenate_datasets([non_elo_ds["train"],non_elo_ds["dev"],non_elo_ds["test"]])
        config["non_elo"] = config_nonelo
    print("non_elo_ds:",non_elo_ds)
    
    ds = datasets.concatenate_datasets([non_elo_ds,elo_ds])
    print("all",ds)
    if args.debias:
        ds = debias(ds)
        print("after debias:", ds)
    if args.save_as_ds:
        ds = split_dataset(ds)
        ds.save_to_disk(args.save_base_path+"/merged_dataset")
    else:
        df = ds.to_pandas()
        responses =[[r["text"] for r in df.loc[i]["responses"]] for i in range(df.shape[0])]
        df["preference"] = responses
        df.rename(columns={"source":"task"},inplace=True)
        df = df[["query","preference","task"]]
        df.to_json(args.save_base_path+"/all_data.jsonl",orient="records",lines=True,force_ascii=False)
    with open(args.save_base_path+'/dataset_config.json', 'w') as json_file:
        json.dump(config, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = set_args(parser)
    args = parser.parse_args()
    main(args)