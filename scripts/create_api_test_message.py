import json
import random
import re


def load_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path,'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_raw_msg(path, roomName):
    # 本地读取数据
    with open(path) as f:
        txt = f.read()
    # 分割文本
    msgs = txt.split("--------分割线--------")
    data = []
    pub_dt = None
    for msg in  msgs:
        msg = msg.strip()
        # 过滤不需要的文本
        if msg.startswith("查看更多消息 : ") or msg.startswith("SYS"):
            m = re.search("\d+年\d+月\d+", msg)
            if m:
                pub_dt = m.group().replace("年","-").replace("月","-")
            continue
        idx = msg.find(":")
        publisher = msg[:idx-1]
        content = msg[idx+1:].lstrip()
        item = dict(roomName=roomName,
                    publisher=publisher,
                    content=content,
                    publishDate=pub_dt,
                    crawlTime="2023-11-08"
                    )
        data.append(item)
    return data


def create_rela_data():
    path1 = "data/origin_data/多多碰票群_202311081940.txt"
    tmp1 = read_raw_msg(path1,roomName="多多碰票群")
    path2 = "data/origin_data/惠鑫兑银行交流群_202311081949.txt"
    tmp2 = read_raw_msg(path1,roomName="惠鑫兑银行交流群")

    tmp = tmp1+tmp2
    sample = random.sample(tmp, 1000)
    save_json(sample, "output/test_rela_api_data.json")   


def create_ner_data():
    data = load_json("../billnerre/data/corpus_msg/dev_processed.json")
    tmp = []
    for item in data:
        new_item = dict(post_data=dict(content=item["text"]),response=dict(preds=item["tag"].split()))
        tmp.append(new_item)
    save_json(tmp, "output/test_ner_api_data.json")


if __name__ == "__main__":
    create_rela_data()
    create_ner_data()