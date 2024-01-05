import re
import json
from collections import defaultdict, namedtuple, Counter

import pandas as pd

from utils.file_io import load_json, save_json
from relation import split as Spliter
from relation.extractor import ReExtractor


LabelItem = namedtuple("LabelItem", ["start","end","text","label"])




def print_json_format(x):
    print(json.dumps(x, indent=2, ensure_ascii=False))






################################ test #####################

def load_test_data():
    path = "../corpus/Step2_已标注语料/final/WJQ_0104_final_2799.json"
    data = load_json(path)
    # 预处理数据格式
    for item in data:
        for label_item in item.get("label",[]):
            # 更名
            label_item['label_name'] = label_item['labels'][0]
            label_item.pop('labels')
        # 更名
        item["labels"] = item.get("label",[])
        if "label" in item:
            item.pop("label")
    return data 


def test_split_pub_vs_msg():
    data = load_test_data()
    tmp = []
    for i,item in enumerate(data):
        text = item["text"]
        labels = item["labels"]
        if len(labels) == 0:
            continue

        # 分割
        publisher_text, publisher_labels, msg_text, msg_labels = Spliter.split_publisher_vs_msg(text, labels)

        pub_item = dict()
        pub_item['publisher'] = publisher_text
        if len(publisher_labels) == 0:
            pub_item['org'] = None 
            pub_item['label'] = None
        else:
            pub_item['org'] = publisher_labels[0]['text']
            pub_item['label'] = publisher_labels[0]['label_name']
        tmp.append(pub_item)
    
    path = "cache/test_split_pub_vs_msg_rst.xlsx"
    pd.DataFrame(tmp).to_excel(path, index=False)
    print("结果请查看文件：%s"%path)


def test_split_msg():
    data = load_test_data()
    tmp = []
    for i,item in enumerate(data):
        text = item["text"]
        labels = item["labels"]
        if len(labels) == 0:
            continue

        # 分割
        publisher_text, publisher_labels, msg_text, msg_labels = Spliter.split_publisher_vs_msg(text, labels)
        items = Spliter.split_msg(msg_text, msg_labels)

        splits = [item["text"] for item in items]
        split_labels = [item["labels"] for item in items]
        tmp.append(dict(raw_text=msg_text, splits=splits))


        # _txt = msg_text.replace("托收","托s").replace("买卖","mm")
        # if _txt.count("买") + _txt.count("卖") + _txt.count("出") + _txt.count("收")  > 1 :
            # tmp.append(dict(raw_text=msg_text, splits=splits,labels=split_labels))


    path =  "cache/test_split_msg_rst.json"
    save_json(tmp, path)
    print("结果请查看文件：%s"%path)    


def test_simple_rule():
    data = load_test_data()
    cache = []
    cache2 = []
    path =  "cache/simple_rule_extract_rst.json"

    re_extractor = ReExtractor()
    re_extractor.add_rule("(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{票据期限}{承兑人}，{贴现人}直?贴(?=$|[^\u4e00-\u9fa5])")
    re_extractor.add_rule("(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{票据期限}{贴现人}贴{承兑人}{金额}")
    re_extractor.add_rule("(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{票据期限}{贴现人}直?贴{承兑人}")
    re_extractor.add_rule("(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{票据期限}{承兑人}\s*?({贴现人}直?贴)*")
    re_extractor.add_rule("(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{票据期限}{承兑人}")
    re_extractor.add_rule("(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{票据期限}")
    re_extractor.add_rule("(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{承兑人}{票据期限}{金额}")
    re_extractor.add_rule("(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{承兑人}")
    re_extractor.add_rule("(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{贴现人}直贴{票据期限}{承兑人}")  
    re_extractor.add_rule("(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{金额}{票据期限}{贴现人}直贴{承兑人}")
    re_extractor.add_rule("(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{贴现人}贴{票据期限}{承兑人}")
    re_extractor.add_rule("(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{金额}{票据期限}{承兑人}")
    re_extractor.add_rule("(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{贴现人}贴{承兑人}{票据期限}票?{金额}")
    re_extractor.add_rule("(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{金额}{票据期限}{贴现人}贴{承兑人}")
    re_extractor.add_rule("(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?我行{贴现人}贴{票据期限}高价{承兑人}")
    re_extractor.add_rule("(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{票据期限}{承兑人}[，,\s]{{0,5}}{金额}")


    tmp1 = []
    tmp2 = []
    tmp3 = []
    tmp4 = []

    for i,item in enumerate(data):
        text = item["text"]
        labels = item["labels"]
        if len(labels) == 0:
            continue

        # 提取发布者所属机构
        org_labels = filter(lambda x:x["label_name"]=="发布者所属机构",labels)
        orgs = list(map(lambda x:x["text"],org_labels))
        if len(orgs) > 0:
            # org = max(orgs, key=lambda x:len(x))  # 取字符长度最长的发布者所属机构
            org = orgs[-1]  # 取最后一个
        else:
            org = None
        org_item = {"发布者所属机构":org}
        # org_item = org_extractor.extract(labels)

        ## 预处理
        # 分割: 发布者信息 与 消息体内容
        publisher_text, publisher_labels, msg_text, msg_labels = Spliter.split_publisher_vs_msg(text, labels)
        # 剔除发布者所属机构
        msg_labels = list(filter(lambda x:x["label_name"]!="发布者所属机构",msg_labels))
        # 切分消息内容
        items = Spliter.split_msg(msg_text, msg_labels)

        last_txn_dir = None
        p_txn = re.compile('[^：:、\d]{0,2}(?<!卖|买|托)(?P<交易方向>出|收|卖断|买断)\D{0,2}')
        for item in items:
            sub_text, sub_labels = item['text'], item["labels"]
            label_names = [item["label_name"] for item in sub_labels]
            # 若无标签，则跳过
            if len(label_names) == 0:
                continue
            cnt = Counter(label_names)
            cnt = {k:v for k,v in cnt.items() if v > 1}  # 重复出现的标签类型及其次数
            # 若标签唯一，则所有标签归为一个完整的要素体
            # if len(label_names) == len(set(label_names)):
            if len(cnt) == 0:
                rst_item = {item["label_name"]:item["text"] for item in sub_labels}
                if len(rst_item) > 0:
                    rst_item.update(org_item)
                m = p_txn.search(sub_text)
                if m:
                    rst_item.update(m.groupdict())
                tmp1.append(dict(text=sub_text,rst=rst_item,raw_text=text))
            elif len(cnt) == 1 and "承兑人" in cnt:
                # 若只是承兑人出现重复
                common_item = {item["label_name"]:item["text"] for item in sub_labels if item["label_name"] != "承兑人"}
                rst_item = dict(common_item)
                rst_item["承兑人"] = ",".join([x["text"] for x in sub_labels if x["label_name"]=="承兑人"])
                if len(rst_item) > 0:
                    rst_item.update(org_item)
                m = p_txn.search(sub_text)
                if m:
                    rst_item.update(m.groupdict())
                tmp2.append(dict(text=sub_text,rst=rst_item,raw_text=text))
            elif len(cnt) == 1 and "票据期限" in cnt:
                rst_item= {item["label_name"]:item["text"] for item in sub_labels if item["label_name"] != "票据期限"}
                rst_item["票据期限"] = ",".join([x["text"] for x in sub_labels if x["label_name"]=="票据期限"])
                if len(rst_item) > 0:
                    rst_item.update(org_item)
                m = p_txn.search(sub_text)
                if m:
                    rst_item.update(m.groupdict())
                tmp4.append(dict(text=sub_text,rst=rst_item,raw_text=text))
                # cache2.append(dict(text=sub_text, labels=sub_labels))
            else:
                # TODO
                rst_item = re_extractor.extract(sub_text, sub_labels)
                if len(rst_item) > 0:
                    rst_item.update(org_item)
                    # tmp2.append(dict(text=sub_text,rst=rst_item2,labels=sub_labels))
                    tmp3.append(dict(text=sub_text,rst=rst_item,raw_text=text))
                else:
                    cache.append(dict(text=sub_text))
                
            # if len(rst_item) > 0:
            #     rst_item.update(org_item)


            
            # tmp.append(dict(text=sub_text,rst=rst_item,raw_text=text))
        
                
            # 若
    save_json(tmp1,  "cache/tmp1.json")
    save_json(tmp2,  "cache/tmp2.json")
    save_json(tmp3,  "cache/tmp3.json")
    save_json(tmp4,  "cache/tmp4.json")
    save_json(cache, "cache/not_unique_msg.json")
    save_json(cache2, "cache/cache2.json")
    print("结果请查看文件：%s"%path)



 
if __name__ == "__main__":
    # test_split_pub_vs_msg()
    # test_split_msg()
    test_simple_rule()
    # print(load_test_data()[0])