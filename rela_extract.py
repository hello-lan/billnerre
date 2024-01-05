import re
from collections import Counter

from relation import split as Spliter
from utils.file_io import load_json,save_json
from relation import MultiSubjectExtractor



mulisubject_extractor = MultiSubjectExtractor()
mulisubject_extractor.add_rule("{票据期限}{承兑人}")
mulisubject_extractor.add_rule("{票据期限}{承兑人}{金额}")
mulisubject_extractor.add_rule("{票据期限}{贴现人}贴{承兑人}")
mulisubject_extractor.add_rule("{票据期限}{贴现人}贴{承兑人}{金额}")
mulisubject_extractor.add_rule("{承兑人}{金额}")
mulisubject_extractor.add_rule("{承兑人}{票据期限}{金额}")
mulisubject_extractor.add_rule("{贴现人}贴{承兑人}{金额}")
mulisubject_extractor.add_rule("{金额}{票据期限}{承兑人}")
mulisubject_extractor.add_rule("{金额}{承兑人}")
mulisubject_extractor.add_rule("{利率}{承兑人}")



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
                method = 1
                rst_items = [rst_item]
            # case 2:若只是`承兑人`出现重复
            elif len(cnt) == 1 and "承兑人" in cnt:
                rst_item = {item["label_name"]:item["text"] for item in sub_labels if item["label_name"] != "承兑人"}
                rst_item["承兑人"] = ",".join([x["text"] for x in sub_labels if x["label_name"]=="承兑人"])
                method = 2
                rst_items = [rst_item]
            # case 3:若只是`票据期限`出现重复
            elif len(cnt) == 1 and "票据期限" in cnt:
                rst_item= {item["label_name"]:item["text"] for item in sub_labels if item["label_name"] != "票据期限"}
                rst_item["票据期限"] = ",".join([x["text"] for x in sub_labels if x["label_name"]=="票据期限"])
                method = 3
                rst_items = [rst_item]
            else:
                # TODO
                rst_items = mulisubject_extractor.extract(sub_text, sub_labels)
                method=4

            if len(rst_items) > 0:
                txn_dir = self.extract_txn_dir(sub_text)
                for rst in rst_items:
                    rst.update(txn_dir)
                    rst.update(org_item)
            result.append(dict(text=sub_text,output=rst_items,raw_text=text,method=method))
        return result



def load_test_data():
    path = "../corpus/Step2_已标注语料/final/WJQ_0104_final_2799.json"
    data = load_json(path)
    # 预处理数据格式
    for item in data:
        txt = item["text"]
        for label_item in item.get("label",[]):
            # 更名
            start, end = label_item["start"],label_item["end"]
            label_item["text"] = txt[start:end]
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
        # print(i,item["text"])
        text = item["text"]
        labels = item["labels"]
        rst = extractor.extract_relation(text, labels)
        tmp.extend(rst)
    
    tmp_01 = filter(lambda x: x["method"]==1, tmp)
    tmp_02 = filter(lambda x: x["method"]==2, tmp)
    tmp_03 = filter(lambda x: x["method"]==3, tmp)
    tmp_04 = filter(lambda x: x["method"]==4, tmp)

    path01 = "cache/extract_test_1.json"
    path02 = "cache/extract_test_2.json"
    path03 = "cache/extract_test_3.json"
    path04 = "cache/extract_test_4.json"
    save_json(list(tmp_01), path01)
    save_json(list(tmp_02), path02)
    save_json(list(tmp_03), path03)
    save_json(list(tmp_04), path04)


    


if __name__ == "__main__":
    test()