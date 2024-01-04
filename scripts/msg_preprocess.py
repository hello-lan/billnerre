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

def toBIOS_v2(anno_dat):
    tmp = []
    for item in anno_dat:
        txt = item["text"]
        tags = ['O'] * len(txt)
        try:
            labels = item['label']
        except:
            labels = []
        for label_info in labels:
            # 修正标注时边缘多标了空白符号的情况
            ent_text = label_info["text"]
            ent_text_1 = ent_text.lstrip()
            ent_text_2 = ent_text.rstrip()
            delta_1 = len(ent_text) - len(ent_text_1)
            delta_2 = len(ent_text) - len(ent_text_2)

            start = label_info["start"] + delta_1
            end = label_info["end"] - delta_2
            # start = label_info["start"]
            # end = label_info["end"]
            # if delta_1 + delta_2 > 0:
            #     print("init: %s"%(txt[start:end]),(start, end))
            #     start += delta_1
            #     end -= delta_2
            #     print("final:%s"%(txt[start:end]),(start, end))

            entity = label_info["labels"][0]
            if entity in ("承接业务",):
                continue
            if start == (end -1):
                tags[start] = "S-%s"%entity
            else:
                B = "B-%s" % entity
                I = "I-%s" % entity
                tags[start] = B 
                tags[start+1: end] = [I] * (end-start-1)
        rst = dict(text=txt, tag=" ".join(tags))
        tmp.append(rst)
    return tmp



def main_preprocess(data_dir):
    processed_data= []
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir,fname)
        data = load_json(path)
        for item in toBIOS_v2(data):
            item["source"] = fname
            processed_data.append(item)

    i = int(len(processed_data) * 0.8)
    shuffle(processed_data)
    data_train = processed_data[:i]
    data_dev = processed_data[i:]

    with open('../data/corpus_msg/train_processed.json','w',encoding='utf-8') as f:
        json.dump(data_train, f, indent=2, ensure_ascii=False)

    with open('../data/corpus_msg/dev_processed.json','w',encoding='utf-8') as f:
        json.dump(data_dev, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    # main_preprocess("msg")
    # main_preprocess("wechat_msg")
    main_preprocess("../../corpus/Step2_已标注语料/final")