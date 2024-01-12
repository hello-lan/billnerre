import re
from collections import Counter

from utils import split as Spliter
from utils.file_io import load_json,save_json
from relation.extractor import (MultiSubjectExtractor, CombineExtractor,
                                MulitDueExtractor)


def extract_publisher_org(labels):
    """抽取发布者所属机构
    """
    org_labels = filter(lambda x:x["label_name"]=="发布者所属机构",labels)
    orgs = list(map(lambda x:x["text"],org_labels))
    if len(orgs) > 0:
        org = orgs[-1]  # 取最后一个
    else:
        org = None
    item = {"发布者所属机构":org}
    return item


def extract_trading_direction(text):
    """ 抽取交易方向
    """
    p = re.compile('[^：:、\d]{0,2}(?<!卖|买|托)(?P<交易方向>出|收|卖断|买断)\D{0,2}')
    m = p.search(text)
    if m:
        return m.groupdict()
    else:
        return dict(交易方向=None)



class RelationExtractor:
    """ 关系抽取器
    """
    def __init__(self,multi_sub_ext, multi_due_ext):
        self.multi_subject_extractor = multi_sub_ext
        self.multi_due_extrator  = multi_due_ext
       
    def extract_relation(self, text, labels):
        """关系抽取"""
        result = []
        if len(labels) == 0:
            return result
        ## step1: 提取发布者所属机构
        org_item = extract_publisher_org(labels)
        ## step2: 预处理
        # 分割: 发布者信息 与 消息体内容
        publisher_text, publisher_labels, msg_text, msg_labels = Spliter.split_publisher_vs_msg(text, labels)
        # 剔除发布者所属机构
        msg_labels = list(filter(lambda x:x["label_name"]!="发布者所属机构",msg_labels))
        # 切分消息内容
        items = Spliter.split_msg(msg_text, msg_labels)
    
        for item in items:
            sub_text, sub_labels = item['text'], item["labels"]
            label_names = [item["label_name"] for item in sub_labels]
            # 若无标签，则跳过
            if len(label_names) == 0:
                continue
            cnt = Counter(label_names)
            cnt = {k:v for k,v in cnt.items() if v > 1}  # 重复出现的标签类型及其出现次数
            # case 1:若标签唯一，则所有标签归为一个完整的要素体
            if len(cnt) == 0:
                rst_items = CombineExtractor().extract(sub_text, sub_labels)
                method = 1
            # case 2:若只是`承兑人`出现重复
            elif len(cnt) == 1 and "承兑人" in cnt:
                rst_items = CombineExtractor(multival_label="承兑人").extract(sub_text, sub_labels)
                method = 2
            # case 3:若只是`票据期限`出现重复
            elif len(cnt) == 1 and "票据期限" in cnt:
                # rst_items = CombineExtractor(multival_label="票据期限").extract(sub_text, sub_labels)
                rst_items = self.multi_due_extrator.extract(sub_text, sub_labels)
                method = 3
            elif len(cnt) == 1 and '贴现人' in cnt:
                rst_items = CombineExtractor(multival_label="贴现人").extract(sub_text, sub_labels)
                method = 5
            else:
                # TODO
                rst_items = self.multi_subject_extractor.extract(sub_text, sub_labels)
                for it in rst_items:
                    it.pop("matched_info")
                method=4

            if len(rst_items) > 0:
                txn_dir = extract_trading_direction(sub_text)   # 抽取交易方向
                for rst in rst_items:
                    rst.update(org_item)
                    if "交易方向" not in rst:
                        rst.update(txn_dir)
            result.append(dict(text=sub_text,output=rst_items,raw_text=text,method=method))
        return result
    
    @classmethod
    def create_rela_extrator(cls):
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
    
        special_muli_due_pattern = [
            "(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{票据期限1}{承兑人1}和{票据期限2}票",
            "(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{票据期限1}{承兑人1}\s*或者{票据期限2}",
        ]
        multi_due_extractor = MulitDueExtractor(special_muli_due_pattern)
    
        rela_extractor = cls(mulisubject_extractor,multi_due_extractor)
        return rela_extractor


def load_test_data():
    path = "../corpus/Step2_已标注语料/final/WJQ_0104_final_2799.json"
    data = load_json(path)
    #
    new_data = []
    # 预处理数据格式
    for item in data:
        txt = item["text"]
        label_items = []
        for label_item in item.get("label",[]):
            start, end = label_item["start"],label_item["end"]
            label_text = txt[start:end]
            label_name = label_item['labels'][0]
            new_label_item=dict(start=start,
                                end=end,
                                text=label_text,
                                label_name=label_name)
            
            label_items.append(new_label_item)
        new_item = dict(text=txt,labels=label_items)
        new_data.append(new_item)
    return new_data 


def test():
    rela_extractor = RelationExtractor.create_rela_extrator()
    # 加载数据
    data = load_test_data()
    tmp = []
    for i,item in enumerate(data):
        text = item["text"]
        labels = item["labels"]
        rst = rela_extractor.extract_relation(text, labels)
        tmp.extend(rst)
    
    tmp_01 = filter(lambda x: x["method"]==1, tmp)
    tmp_02 = filter(lambda x: x["method"]==2, tmp)
    tmp_03 = filter(lambda x: x["method"]==3, tmp)
    tmp_04 = filter(lambda x: x["method"]==4, tmp)
    tmp_05 = filter(lambda x: x["method"]==5, tmp)


    path01 = "cache/extract_test_1.json"
    path02 = "cache/extract_test_2.json"
    path03 = "cache/extract_test_3.json"
    path04 = "cache/extract_test_4.json"
    path05 = "cache/extract_test_5.json"
    save_json(list(tmp_01), path01)
    save_json(list(tmp_02), path02)
    save_json(list(tmp_03), path03)
    save_json(list(tmp_04), path04)
    save_json(list(tmp_05), path05)



if __name__ == "__main__":
    test()