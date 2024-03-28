import re
import random
import pandas as pd
from datasets import DatasetDict,Dataset

def punc_repetition(text):
    # 随机重复某个标点符号
    puncs = ['。',"，","\\n","\\t",".",",",":","：","\n","\t"]
    for punc in random.sample(puncs,len(puncs)):
        if punc in text:
            times = random.choice([10,20,30])
            text = text.replace(punc,punc*times)
            break
    return text

def text_repetition(text):
    # 随机重复某个句子
    texts = re.split('(。|\\n)', text)
    texts = [texts[i]+texts[i+1] for i in range(0,len(texts)-1,2)] if len(texts)>1 else texts
    index = random.choice(range(len(texts)))
    times = random.choice([3,5,10])
    texts[index] = texts[index]*times
    text = "".join(texts)
    return text

def delete_punc(text):
    # 删除文本中所有标点符号
    punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    text = re.sub(r"[%s]+" %punc, "",text)
    text = re.sub(r"\n", "",text)
    text = re.sub(r"\t", "",text)
    return text

def void_text(text):
    # 生成各类空白样本
    text = random.choice([""," ","\n","\t"])
    text = text*random.choice([1,3,10,20])
    return text

def shuffle_query_response(query_list,response_list):
    # 打乱query和response的顺序
    response_list=random.sample(response_list,len(response_list))
    return query_list,response_list

def change_number(text):
    # 随机改变文本中的数字
    numbers = re.findall(r"\d+\.?\d*",text)
    if len(numbers)==0:
        return None
    number_org = random.choice(numbers)
    number = float(number_org)
    add = random.choice(range(min(-int(number),-10),max(int(number),10)))
    number_new = str(number+add).replace(".0","")
    text = text.replace(number_org,number_new)
    return text

def split_dataset(ds):
    ds = ds.train_test_split(test_size=0.1)
    ds_holdout = ds['test'].train_test_split(test_size=0.5)
    ds_save = DatasetDict()
    ds_save['train']=ds['train']
    ds_save['dev']=ds_holdout['train']
    ds_save['test']=ds_holdout['test']
    return ds_save

def debias(ds):
    # 
    def change_newline(example):
        q_0 = example["responses"][0]["text"]
        q_1 = example["responses"][-1]["text"]
        if "\n\n" in q_0:
            r = random.random()
            if r < 0.2:
                q_0 = q_0.replace("\n\n", "\n")
                example["responses"][0]["text"] = q_0
        if "。\n" in q_1 and "。\n\n" not in q_1:
            r = random.random()
            if r < 0.2:
                q_1 = q_1.replace("。\n", "。\n\n")
                example["responses"][-1]["text"] = q_1
        return example

    ds = ds.map(change_newline)
    print(ds)
    df = ds.to_pandas()
    clip_rate=2
    responses_lens = []
    for i in range(df.shape[0]):
        responses_lens.append([len(r["text"]) for r in df.loc[i]["responses"]])
    avg_diff = []
    for lens in responses_lens:
        avg = [lens[i]-lens[i+1] for i in range(len(lens)-1)]
        avg = sum(avg)/len(avg)
        avg_diff.append(avg)
    df["avg_lens_diff"] = avg_diff
    neg_ranges = [(-20000,-1000),(-1000,-250),(-250,-100),(-100,-50),(-50,0)]
    pos_ranges = [(1000,20000),   (250,1000),  (100,250),  (50,100),  (0,50)]
    all_data = []
    for pos_rs,neg_rs in zip(pos_ranges,neg_ranges):
        # (0,50)之间的样本不需要clip
        df_tmp_pos = df[(df['avg_lens_diff']>=pos_rs[0]) & (df['avg_lens_diff']<pos_rs[1])]
        df_tmp_neg = df[(df['avg_lens_diff']>=neg_rs[0]) & (df['avg_lens_diff']<neg_rs[1])]
        print(f'BEFORE: range: {pos_rs}, count: {len(df_tmp_pos)},range: {neg_rs}, count: {len(df_tmp_neg)}')
        if pos_rs[0]!=0:
            neg_count = len(df_tmp_neg)
            df_tmp_pos = df_tmp_pos.sample(min(int(neg_count*clip_rate),len(df_tmp_pos)))
        print(f'AFTER: range: {pos_rs}, count: {len(df_tmp_pos)},range: {neg_rs}, count: {len(df_tmp_neg)}')
        all_data.append(df_tmp_pos.reset_index(drop=True))
        all_data.append(df_tmp_neg.reset_index(drop=True))
    df = pd.concat(all_data, ignore_index=True).reset_index(drop=True)
    ds = Dataset.from_pandas(df)
    return ds