import os
import json
from random import shuffle


def load_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data


def covert_to_bios(anno_dat):
    bios_data = []
    for item in anno_dat:
        txt = item["text"]
        tags = ['O'] * len(txt)
        labels = item.get("label",[])
        for label_info in labels:
            # 修正标注时边缘多标了空白符号的情况
            ent_text = label_info["text"]
            ent_text_1 = ent_text.lstrip()
            ent_text_2 = ent_text.rstrip()
            delta_1 = len(ent_text) - len(ent_text_1)
            delta_2 = len(ent_text) - len(ent_text_2)

            start = label_info["start"] + delta_1
            end = label_info["end"] - delta_2
            # 过滤掉标签“承接业务”
            entity = label_info["labels"][0]
            if entity in ("承接业务",):
                continue
            # BIOS标注
            if start == (end -1):
                tags[start] = "S-%s"%entity
            else:
                B = "B-%s" % entity
                I = "I-%s" % entity
                tags[start] = B 
                tags[start+1: end] = [I] * (end-start-1)
        rst = dict(text=txt, tag=" ".join(tags))
        bios_data.append(rst)
    return bios_data



def main_preprocess(src_dir, dst_dir):
    processed_data= []
    for fname in os.listdir(src_dir):
        path = os.path.join(src_dir,fname)
        data = load_json(path)
        for item in covert_to_bios(data):
            item["source"] = fname
            processed_data.append(item)
    # 划分数据集
    i = int(len(processed_data) * 0.8)
    shuffle(processed_data)
    data_train = processed_data[:i]
    data_dev = processed_data[i:]

    # 保存
    path_train = os.path.join(dst_dir,"train_processed.json")
    path_dev = os.path.join(dst_dir,"dev_processed.json")

    with open(path_train,'w',encoding='utf-8') as f:
        json.dump(data_train, f, indent=2, ensure_ascii=False)

    with open(path_dev,'w',encoding='utf-8') as f:
        json.dump(data_dev, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    # main_preprocess("data/annotated_data","output")
    main_preprocess("data/annotated_data","../billnerre/data/corpus_msg")