import re
from collections import Counter

from relation import split as Spliter
from utils.file_io import load_json,save_json
from relation.extractor import MultiSubjectExtractor, CombineExtractor,TradingDirectionExtractor,PublisherOrgExtactor



class RelationExtractor:
    def __init__(self):
        self.pulisher_extractor = None
        self.trading_direction_extractor = None
        self.multi_subject_extractor = None
    
    def _check_extrators(self):
        tmp = []
        if self.pulisher_extractor is None:
            tmp.append("publisher_extractor")
        if self.trading_direction_extractor is None:
            tmp.append('trading_direction_extractor')
        if self.multi_subject_extractor is None:
            tmp.append("multi_subject_extractor")
        
        if len(tmp) > 0:
            s = "`" + "`,`".join(tmp) + "`" 
            info = "Attribute %s haven't been added, please call `add_xxx_extractor` method to add these Atrributers" %s
            raise AttributeError(info)

    def add_publisher_org_extractor(self, extractor):
        self.pulisher_extractor = extractor

    def add_trading_direction_extractor(self, extractor):
        self.trading_direction_extractor = extractor

    def add_multi_subject_extractor(self, extractor):
        self.multi_subject_extractor = extractor

    def extract_org(self, labels):
        """ 抽取发布者所属机构
        """
        item = self.pulisher_extractor.extract("", labels)
        return item
    
    def extract_txn_dir(self, text, labels=[]):
        """ 抽取交易方向
        """
        item = self.trading_direction_extractor.extract(text, labels)
        return  item
    
    def extract_multi_subject(self, text, labels):
        item = self.multi_subject_extractor.extract(text, labels)
        return item

    def extract_relation(self,text, labels):
        self._check_extrators()

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
                rst_item = CombineExtractor().extract(sub_text, sub_labels)
                method = 1
                rst_items = [rst_item]
            # case 2:若只是`承兑人`出现重复
            elif len(cnt) == 1 and "承兑人" in cnt:
                rst_item = CombineExtractor(multival_label="承兑人").extract(sub_text, sub_labels)
                method = 2
                rst_items = [rst_item]
            # case 3:若只是`票据期限`出现重复
            elif len(cnt) == 1 and "票据期限" in cnt:
                rst_item = CombineExtractor(multival_label="票据期限").extract(sub_text, sub_labels)
                method = 3
                rst_items = [rst_item]
            else:
                # TODO
                rst_items = self.extract_multi_subject(sub_text, sub_labels)
                method=4

            if len(rst_items) > 0:
                txn_dir = self.extract_txn_dir(sub_text, sub_labels)
                for rst in rst_items:
                    rst.update(txn_dir)
                    rst.update(org_item)
            result.append(dict(text=sub_text,output=rst_items,raw_text=text,method=method))
        return result



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


def create_extractor():
    ## 初始化
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

    rela_extractor = RelationExtractor()
    rela_extractor.add_multi_subject_extractor(mulisubject_extractor)
    rela_extractor.add_publisher_org_extractor(PublisherOrgExtactor())
    rela_extractor.add_trading_direction_extractor(TradingDirectionExtractor())
    return rela_extractor


def test():
    rela_extractor = create_extractor()
    # 加载数据
    data = load_test_data()
    tmp = []
    for i,item in enumerate(data):
        # print(i,item["text"])
        text = item["text"]
        labels = item["labels"]
        rst = rela_extractor.extract_relation(text, labels)
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