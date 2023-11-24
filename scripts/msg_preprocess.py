import os
import json
from random import shuffle
from collections import Counter, defaultdict

import pandas as pd


def load_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data


def toBIOS(data):
    tmp = []
    for item in data:
        text = item["text"]
        tag = ["O"] * len(list(text))
        for info in item.get("label",[]):
            start = info['start']
            end = info['end']
            lab = info["labels"][0]
            if start == (end -1):
                tag[start] = "S-" + lab
            else:
                tag[start] = "B-" + lab
                tag[start+1:end] = ["I-" + lab] * (end-start-1)
        rst = dict(text=text, tag=" ".join(tag))
        tmp.append(rst)
    return tmp


def main_preprocess(data_dir):
    processed_data= []
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir,fname)
        data = load_json(path)
        processed_data.extend(toBIOS(data))

    i = int(len(processed_data) * 0.8)
    shuffle(processed_data)
    data_train = processed_data[:i]
    data_dev = processed_data[i:]

    with open('../data/wechat_msg/train_processed.json','w',encoding='utf-8') as f:
        json.dump(data_train, f, indent=2)

    with open('../data/wechat_msg/dev_processed.json','w',encoding='utf-8') as f:
        json.dump(data_dev, f, indent=2)


def main_statistic(data_dir):
    """统计标注实体"""
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir,fname)
        data = load_json(path)
        ent_of_id = defaultdict(set)
        counter = Counter()
        sheet = fname.split(".")[0]
        for item in data:
            id_ = item["id"]
            if "label" not in item:
                continue
            for e in item["label"]:
                ent = e['text']
                ent_of_id[ent].add(id_)
                counter[ent] += 1
        df = pd.Series(counter).to_frame(name='freq')
        df["txt_ids"] = df.index.map(ent_of_id)
        
        df.sort_values(by="freq",ascending=False).to_excel(f'../cache/{sheet}.xlsx')



publishers = ["🌽米修","鹭岛小丫🐯","🍀 ","二任一点都不愣🌚","MarkL🐴"]



if __name__ == "__main__":
    # main_statistic("wechat_msg")
    # main_preprocess("msg")
    main_preprocess("wechat_msg")