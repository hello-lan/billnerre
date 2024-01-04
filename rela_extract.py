import re
from collections import Counter

from relation import split as Spliter
from utils.file_io import load_json,save_json



class RelationExtractor:
    def __init__(self):
        self.txn_pattern = re.compile('[^：:、\d]{0,2}(?<!卖|买|托)(?P<交易方向>出|收|卖断|买断)\D{0,2}')

    def extract_org(self, labels):
        """ 抽取发布者所属机构
        """
        org_labels = filter(lambda x:x["label_name"]=="发布者所属机构",labels)
        orgs = list(map(lambda x:x["text"],org_labels))
        if len(orgs) > 0:
            # org = max(orgs, key=lambda x:len(x))  # 取字符长度最长的发布者所属机构
            org = orgs[-1]  # 取最后一个
        else:
            org = None
        org_item = {"发布者所属机构":org}
        return org_item
    
    def extract_txn_dir(self, text):
        """ 抽取交易方向
        """
        m = self.txn_pattern.search(text)
        return  m.groupdict() if m else dict(交易方向=None)

    def extract_relation(self,text, labels):
        result = []
        if len(labels) == 0:
            return result
        ## step1: 提取发布者所属机构
        org_item = self.extract_org(labels)
        ## step2: 预处理
        # 分割: 发布者信息 与 消息体内容
        publisher_text, publisher_labels, msg_text, msg_labels = Spliter.split_publisher_vs_msg(text, labels)
        # 剔除发布者所属机构
        msg_labels = list(filter(lambda x:x["label_name"]!="发布者所属机构",msg_labels))
        # 切分消息内容
        items = Spliter.split_msg(msg_text, msg_labels)
    
        last_txn_dir = None
        for item in items:
            sub_text, sub_labels = item['text'], item["labels"]
            label_names = [item["label_name"] for item in sub_labels]
            # 若无标签，则跳过
            if len(label_names) == 0:
                continue
            cnt = Counter(label_names)
            cnt = {k:v for k,v in cnt.items() if v > 1}  # 重复出现的标签类型及其次数
            # case 1:若标签唯一，则所有标签归为一个完整的要素体
            if len(cnt) == 0:
                rst_item = {item["label_name"]:item["text"] for item in sub_labels}
            # case 2:若只是`承兑人`出现重复
            elif len(cnt) == 1 and "承兑人" in cnt:
                rst_item = {item["label_name"]:item["text"] for item in sub_labels if item["label_name"] != "承兑人"}
                rst_item["承兑人"] = ",".join([x["text"] for x in sub_labels if x["label_name"]=="承兑人"])
            # case 3:若只是`票据期限`出现重复
            elif len(cnt) == 1 and "票据期限" in cnt:
                rst_item= {item["label_name"]:item["text"] for item in sub_labels if item["label_name"] != "票据期限"}
                rst_item["票据期限"] = ",".join([x["text"] for x in sub_labels if x["label_name"]=="票据期限"])
            else:
                # TODO
                continue

            if len(rst_item) > 0:
                txn_dir = self.extract_txn_dir(sub_text)
                rst_item.update(txn_dir)
                rst_item.update(org_item)
                result.append(dict(text=sub_text,output=rst_item,raw_text=text))
        return result



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


def test():
    extractor = RelationExtractor()
    data = load_test_data()
    tmp = []
    for i,item in enumerate(data):
        text = item["text"]
        labels = item["labels"]
        rst = extractor.extract_relation(text, labels)
        tmp.extend(rst)
    path = "cache/extract_test.json"
    save_json(tmp, path)



if __name__ == "__main__":
    test()