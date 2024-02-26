import re
import warnings
from collections import defaultdict, Counter
from abc import ABC, abstractclassmethod

from .util import join



## 抽象基类（接口）：抽取器
class Extractor(ABC):
    @abstractclassmethod
    def extract(self, text, labels):
        pass

    def __str__(self):
        return self.__class__.__name__


class IntegrateExtractor(Extractor):
    """组合关系抽取器，把出现的标签直接组合成一组抽取信息"""
    def __init__(self, multival_label=None):
        self.multival_label = multival_label

    def __str__(self):
        return self.__class__.__name__ + "(multival_label=%s)" % self.multival_label

    def _preprocessing(self, text,labels):
        return text, labels

    def _checking_condition(self, text:str, labels:list):
        """ 使用IntegrateExtractor需要确保除了标签multival_label外，其他标签名唯一
        """
        label_names = [item["label_name"] for item in labels]
        counter = Counter(label_names)
        dumplcates = {k:v for k,v in counter.items() if v > 1}  # 重复出现的标签类型及其出现次数
        if self.multival_label is None:
            return len(dumplcates) == 0
        elif len(dumplcates)==1 and self.multival_label in dumplcates:
            return True
        else:
            return False

    def extract(self, text, labels):
        if not self._checking_condition(text, labels):
            return []
        text_, labels_ = self._preprocessing(text, labels)
        item = {item["label_name"]:item["text"] for item in labels_}
        if self.multival_label is not None:
            multival_label = self.multival_label
            item[multival_label] = [item["text"] for item in labels_ if item["label_name"]==multival_label]
        return [item]
        
    def __str__(self):
        return "{cls_name}(multival_label={arg})".format(cls_name=self.__class__.__name__, arg=self.multival_label)
    

class MultiDueExtractor(Extractor):
    """多个票据期限（其他标签唯一）的关系抽取器
    """
    def __init__(self,special_patterns=None):
        self.integrate_extractor = IntegrateExtractor(multival_label="票据期限")
        self.templates_extractor = TemplateExtractor(special_patterns)

    def extract(self, text, labels):
        rst_1 = self.integrate_extractor.extract(text, labels)
        rst_2 = self.templates_extractor.extract(text, labels)
        return rst_2 if len(rst_1) > 0 and len(rst_2) > 0 else rst_1


class MultiAmtExtractor(IntegrateExtractor):
    """多个金额（其他标签唯一）的关系抽取器
    """
    def __init__(self):
        super().__init__(multival_label="金额")

    def _checking_condition(self, text: str, labels: list):
        cond_1 = super()._checking_condition(text,labels)
        cond_2 = re.search("单张\d+",text)
        return cond_1 and cond_2

    def _preprocessing(self, text, labels):
        amts = [label["text"] for label in labels if label["label_name"]==self.multival_label]
        exp = "|".join(amts)
        price_exp = "单张(%s)"%exp
        prices = re.findall(price_exp, text)
        new_labels = list(filter(lambda x:x["text"] not in prices, labels))
        return text, new_labels


class TemplateExtractor(Extractor):
    """ 模板抽取器
    """
    def __init__(self, templates:list=None):
        if isinstance(templates,list):
            self.templates = templates
        else:
            self.templates = []
        self.template_patterns = []

    def __str__(self):
        return self.__class__.__name__ + "(templates=[...])"

    def add_template(self, template):
        self.templates.append(template)
    
    def add_templates(self, templates:list):
        self.templates.extend(templates)

    @staticmethod
    def _join_entities(label2entities):
        """ 拼接同类实体为为正则表达式形式
        """
        label2entity_express = dict()
        for label, entities in label2entities.items():
            if label == "贴现人":
                express = join(entities, prefix="/?",sufix="贴?")
            elif label == "承兑人":
                # express = join(entities, prefix="[、,，\s及和/+或者]{0,3}",sufix="(?!贴)")
                express = join(entities, prefix="[、,，\s及和/+或者]{0,3}",sufix="(?:小票|大票|大小票)?(?!贴)")
            elif label == "票据期限":
                express = join(entities, prefix="",sufix="[\.\s、，及和至\-+/或者少量]*")
            else:
                express = join(entities)
            label2entity_express[label] = express
        return label2entity_express
    
    def _fill_templates(self, label2entity_express):
        """ 将实体填充模板中对应的位置，并将填充好的模板编译为正则表达式pattern
        """
        base_mapping =  {label:"(?P<{name}>(?:{exp})+)".format(name=label,exp=exp) for label, exp in label2entity_express.items()}
        # 填充模板
        filled_templates = []
        for template in self.templates:
            mapping = dict(base_mapping)
            # 若模板中的标签名存在数字后缀，则补充带数字后缀的命名分组表达式（正则）
            nums = re.findall("(?<!\{)\{.*?(\d+)\}(?!\})", template)
            if len(nums) > 0:
                max_num = max(map(int, nums))
                for label, express in label2entity_express.items():
                    for i in range(1,max_num+1):
                        new_label = label+str(i)
                        mapping[new_label] = "(?P<{name}>{exp})".format(name=new_label,exp=express)
           
            # 填充模板
            try:
                filled_template = template.format_map(mapping)
                filled_templates.append(filled_template)
            except:
                # warnings.warn("template `%s` fill element %s failed"%(r, mapping))
                continue
        # 将填充好的模板转为正则表达式Pattern
        self.template_patterns.clear()   # 清空
        for regexp in filled_templates:
            try:
                p = re.compile(regexp)
            except:
                warnings.warn("regexp `%s` compile failed"%regexp)
                continue
            else:
                self.template_patterns.append(p)

    def _exract_from_templates(self, text):
        items = []
        for p in self.template_patterns:
            m = p.search(text)
            if m:
                items.append(m.groupdict())
        if len(items) > 0:
            best_item = max(items, key=lambda x: len(x))   # 取抽取到的元素数量最多的结果
            return self._assemble_extracted_item(best_item)
        else:
            return []

    @staticmethod  
    def _assemble_extracted_item(item):
        """ 组装模板抽取的结果
        """
        num_p = re.compile("\d+$")
        base_group, num_group = dict(), defaultdict(dict)
        ## 1. 分离标签的后缀数字
        for label, entity in item.items():
            nums = num_p.findall(label)
            if len(nums) == 0:
                base_group[label] = entity
            else:
                num = nums[0]
                new_label = num_p.sub("", label)
                num_group[num][new_label] = entity
        output = []
        ## 2. 填充共有标签实体
        if len(num_group) > 0:
            for group in num_group.values():
                group.update(base_group)
                output.append(group)
        else:
            if len(base_group) > 0:
                output.append(base_group)
        return output
    
    @staticmethod
    def _melt_entities(items, label2entities):
        # prepare regexps
        label2regexp = dict()
        for label_name, entities in label2entities.items():
            ents = sorted(entities,key=lambda x:len(x), reverse=True)  # 按字符数量从多到少排序
            label2regexp[label_name] = "({regexp})".format(regexp="|".join(ents))
        # 
        for item in items:
            for label_name in item.keys():
                if label_name in label2regexp:
                    regexp = label2regexp[label_name]
                    entity = item[label_name]
                    entities = re.findall(regexp,entity)
                    item[label_name] = entities if len(entities) > 1 else entities[0]
        return items


    def extract(self, text, labels):
        label2entities = defaultdict(set)
        for info in labels:
            label_name = info["label_name"]
            entity = info["text"]
            entity = entity.replace("\\",'\\\\')
            label2entities[label_name].add(entity)

        label2express = self._join_entities(label2entities)
        self._fill_templates(label2express)  # 填充模板
        items = self._exract_from_templates(text)   # 模板抽取
        new_items = self._melt_entities(items, label2entities)
        return new_items


class MultiSubjectExtractor(TemplateExtractor):
    def add_rule(self, rule):
        super().add_template(rule)

    def _exract_from_templates(self, text):
        items = []
        for p in self.template_patterns:
            m = p.search(text)
            if m:
                item = m.groupdict()
                sub_text = m.group()
                item_wrapper = dict()
                item_wrapper["item"] = item
                item_wrapper["span"] = m.span()
                item_wrapper["first_entity_start_index"] = min([sub_text.find(s) + m.start() for s in item.values()])  # 提取到第一个的实体开始位置
                items.append(item_wrapper)
        
        if len(items) > 0:
            best_item = max(items, key=lambda x: (-x["first_entity_start_index"], len(x["item"])))
            start, end = best_item["span"]
            new_text = text[:start] + "$"*(end-start)  + text[end:]
            return self._assemble_extracted_item(best_item["item"]) + self._exract_from_templates(new_text)
        else:
            return []
        
    def extract(self, text, labels):
        items = super().extract(text, labels)
        labels = sorted(labels, key=lambda x:x["start"])   # 按索引位置排序
        #case 1 如果没有提取到贴现人（贴现人在最后）
        # discounter_items = list(filter(lambda x:"贴现人" in x, items))
        # # discounter_labels = filter(lambda x:x["label_name"]=="贴现人",labels[-2:])   # 最后两个标签
        # discounter_labels = filter(lambda x:x["label_name"]=="贴现人",labels)   # 最后两个标签
        # discounters = list(map(lambda x:x["text"],discounter_labels))
        # if len(discounter_items)==0 and len(discounters) > 0:
        #     for item in items:
        #         item["贴现人"] = discounters

        # 贴现人在最后
        last_discounter = []
        for label in labels[::-1]:
            if label["label_name"] =="贴现人":
                last_discounter.append(label["text"])
            else:
                break

        if len(last_discounter) > 0:
            last_item = items[-1]
            last_discounter = last_item.get("贴现人",last_discounter)
            for i in range(len(items)-1,-1,-1):
                item = items[i]
                if i == (len(items)-1) or item.get("贴现人") is None:
                    item["贴现人"] = last_discounter
                else:
                    break

        # CASE 2: 第一个抽取抽取项有票据期限，后面的都没有
        pre_duetime = []
        for label in labels:
            if label["label_name"] == "票据期限":
                pre_duetime.append(label["text"])
            else:
                break
        for item in items:
            pre_duetime = item.get("票据期限", pre_duetime)
            item["票据期限"] = pre_duetime
        return items