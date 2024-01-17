import re

from utils import split as Spliter
from utils.file_io import load_json,save_json
from relation.extractor import (MultiSubjectExtractor, MultiAmtExtractor,
                                MultiDueExtractor, IntegrateExtractor, TemplateExtractor)


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


class RelationExtractorManager:
    """ 关系抽取器
    """
    def __init__(self, extractors:list):
        self.rela_extractors = extractors
       
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
            for j, extractor in enumerate(self.rela_extractors):
                rst_items = extractor.extract(sub_text, sub_labels)
                if len(rst_items) > 0:
                    # 补充其他要素
                    txn_dir = extract_trading_direction(sub_text)   # 抽取交易方向
                    for rst in rst_items:
                        rst.update(org_item)  # 添加发布者所属机构
                        if "交易方向" not in rst:
                            rst.update(txn_dir)   # 添加交易方向 
                    # 保存结果
                    method = str(extractor)
                    method = j+1
                    result.append(dict(text=sub_text,labels=sub_labels,output=rst_items,raw_text=text,method=method))
                    # 跳出
                    break
            else:
                pass
        return result
    
    @classmethod
    def create_rela_extrator(cls):
        uniq_subject_extractor = IntegrateExtractor()
        multi_discounter_extractor = IntegrateExtractor("贴现人")
        multi_accptor_extractor = IntegrateExtractor("承兑人")
        multi_amt_extractor = MultiAmtExtractor()

        special_multi_due_pattern = [
            "(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{票据期限1}{承兑人1}和{票据期限2}票",
            "(?<!卖|买|托)【?(?P<交易方向>出|收|买|卖)[.】：\w]{{0,3}}?{票据期限1}{承兑人1}\s*或者{票据期限2}",
        ]
        multi_due_extractor = MultiDueExtractor(special_multi_due_pattern)

        multi_subject_extractor = MultiSubjectExtractor()
        multi_subject_extractor.add_rule("{票据期限}（?{承兑人}）?")
        multi_subject_extractor.add_rule("{票据期限}{利率}{承兑人}")
        multi_subject_extractor.add_rule("{票据期限}{承兑人}{金额}")
        multi_subject_extractor.add_rule("{票据期限}{承兑人}单张{金额}")
        multi_subject_extractor.add_rule("{票据期限}{贴现人}直?贴（?{承兑人}）?")
        multi_subject_extractor.add_rule("{票据期限}{贴现人}贴{承兑人}{金额}")
        multi_subject_extractor.add_rule("{票据期限}{贴现人}贴{承兑人}承兑票{金额}")
        multi_subject_extractor.add_rule("{承兑人}\s*{金额}")
        multi_subject_extractor.add_rule("{承兑人}{票据期限}{金额}")
        multi_subject_extractor.add_rule("{贴现人}贴{承兑人}（?{金额}）?")
        multi_subject_extractor.add_rule("{贴现人}直?贴{承兑人}")
        multi_subject_extractor.add_rule("{金额}{票据期限}{承兑人}")
        multi_subject_extractor.add_rule("{金额}{票据期限}{贴现人}贴{承兑人}")
        multi_subject_extractor.add_rule("{金额}{承兑人}")
        multi_subject_extractor.add_rule("{利率}{承兑人}")
        multi_subject_extractor.add_rule("{承兑人}{利率}")
        multi_subject_extractor.add_rule("{承兑人}{票据期限}") # 票据期限主要为`托收`
        multi_subject_extractor.add_rule("{票据期限}各类票")
        multi_subject_extractor.add_rule("{利率}出{承兑人}{金额}，{贴现人}贴")
        multi_subject_extractor.add_rule("收{票据期限}[；;]")
        multi_subject_extractor.add_rule("{利率}出{票据期限}{贴现人}贴{承兑人}{金额}")
        multi_subject_extractor.add_rule("收{承兑人}、")

        template_extractor = TemplateExtractor()
    
        extractors = [
            uniq_subject_extractor,
            multi_discounter_extractor,
            multi_accptor_extractor,
            multi_amt_extractor,
            multi_due_extractor,
            # template_extractor,
            multi_subject_extractor
        ]

        manager = cls(extractors)
        return manager


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
            if label_name == "承接业务":
                continue
            new_label_item=dict(start=start,
                                end=end,
                                text=label_text,
                                label_name=label_name)
            
            label_items.append(new_label_item)
        new_item = dict(text=txt,labels=label_items)
        new_data.append(new_item)
    return new_data 


def test():
    manager = RelationExtractorManager.create_rela_extrator()
    # 加载数据
    data = load_test_data()
    tmp = []
    for i,item in enumerate(data):
        text = item["text"]
        labels = item["labels"]
        rst = manager.extract_relation(text, labels)
        tmp.extend(rst)
    
    def pop_labels(item):
        item_new = dict(item)
        item_new.pop("labels")
        return item_new
    
    tmp_01 = map(pop_labels,filter(lambda x: x["method"]==1, tmp))
    tmp_02 = map(pop_labels,filter(lambda x: x["method"]==2, tmp))
    tmp_03 = map(pop_labels,filter(lambda x: x["method"]==3, tmp))
    tmp_04 = map(pop_labels,filter(lambda x: x["method"]==4, tmp))
    tmp_05 = map(pop_labels,filter(lambda x: x["method"]==5, tmp))
    tmp_06 = map(pop_labels,filter(lambda x: x["method"]==6, tmp))

    def is_complete(item):
        labels = item["labels"]
        output = item["output"]
        entities_01 = [x["text"] for x in labels]
        entities_02_ = [xx for x in output for xx in x.values()]
        entities_02 = []
        for e in entities_02_:
            if isinstance(e, list):
                entities_02.extend(e)
            elif isinstance(e, str):
                entities_02.append(e)
        y = set(entities_01) - set(entities_02)
        return len(y)  > 0 and item["method"]==6
    
    tmp_07 = map(pop_labels,(filter(is_complete, tmp)))

    path01 = "cache/extract_test_1.json"
    path02 = "cache/extract_test_2.json"
    path03 = "cache/extract_test_3.json"
    path04 = "cache/extract_test_4.json"
    path05 = "cache/extract_test_5.json"
    path06 = "cache/extract_test_6.json"
    path07 = "cache/extract_test_7.json"
    save_json(list(tmp_01), path01)
    save_json(list(tmp_02), path02)
    save_json(list(tmp_03), path03)
    save_json(list(tmp_04), path04)
    save_json(list(tmp_05), path05)
    save_json(list(tmp_06), path06)
    save_json(list(tmp_07), path07)



if __name__ == "__main__":
    test()