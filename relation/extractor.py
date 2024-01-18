import re
import warnings
from collections import defaultdict, Counter
from abc import ABC, abstractclassmethod


def join_entities_as_regexp(entities, prefix='',sufix=''):
    """将多个具有相同标签的实体拼接得到一个正则表达式
    """
    entities = sorted(entities,key=lambda x:len(x), reverse=True)
    _entities = [prefix + ent + sufix for ent in entities]
    regexp = "|".join(_entities)
    return regexp


## 抽象基类（接口）：抽取器
class Extractor(ABC):
    @abstractclassmethod
    def extract(self, text, labels):
        pass

    def __str__(self):
        return self.__class__.__name__
   


class MultiSubjectExtractor(Extractor):
    """ 多要素主体抽取器
    用于单条文本中含有多个要素主体的情况，
    如：
       "收12月国贴大商及1.2月国贴城农、财司电商高价票，欢迎清单报价"
       "出少量9月到期双国股，2月到期双国股"
    """
    def __init__(self, rules=None):
        if isinstance(rules,list):
            self.rules = rules
        else:
            self.rules = []

    def add_rule(self, rule):
        self.rules.append(rule)

    def add_rules(self, rules):
        self.rules.extend(rules)

    def _build_regexp_patterns(self, tags):
        # 1.准备待填充的要素（正则表达式形式）
        mapping = dict()
        for k, v in tags.items():
            if k == "贴现人":
                v = [vv+"贴?" for vv in v]
                val = join_entities_as_regexp(v, prefix="/?")
            elif k == "承兑人":
                val = join_entities_as_regexp(v, prefix="[、,，\s及和/+或者]{0,3}",sufix="(?!贴)")
            elif k == "票据期限":
                val = join_entities_as_regexp(v, prefix="",sufix="[\.\s、，及和至或者\-+/]*")
            else:
                val =  join_entities_as_regexp(v)
            mapping[k] = "(?P<{name}>(?:{exp})+)".format(name=k,exp=val)
        # 2. 填充正则表示式模板
        regexps = []
        for r in self.rules:
            try:
                regexp = r.format_map(mapping)
            except:
                # warnings.warn("rule `%s` fill element %s failed" %(r, mapping))
                continue 
            else:
                regexps.append(regexp)
        # 3. 创建正则表示式对象
        patterns = []
        for regexp in regexps:
            try:
                p = re.compile(regexp)
            except:
                warnings.warn("regexp `%s` compile failed" % regexp)
                continue
            else:
                patterns.append(p)
        return patterns

    def re_extract(self, pattern, text):
        """正则表达式抽取"""
        m = pattern.search(text)
        if m:
            item = m.groupdict()
            # item = {k:v for k,v in item.items() if v is not None}        # 过滤None值
            sub_text = m.group()
            first_ent_idx = min([sub_text.find(s) + m.start() for s in item.values()])  # 提取到第一个的实体开始位置
            item["matched_info"] = dict(pattern=pattern.pattern, span=m.span(), first_ent_idx=first_ent_idx)  # 正则表达式匹配信息
            return item
        else:
            return dict()
        
    def _extract(self, text, re_patterns):
        items = [self.re_extract(p, text) for p in re_patterns]
        items = [item for item in items if len(item) > 0]
        if len(items) > 0:
            # item = max(items, key=lambda x:len(x))  
            item = max(items, key=lambda x:(-x["matched_info"]["first_ent_idx"],len(x)))  # 按元素数量大小，位置前后 排序去第一个
            # 把已经匹配成功要素在原text中剔除（用`$$$$$$$$`替换），再重新放回进行正则抽取
            start, end = item["matched_info"]["span"]
            new_text = text[:start] + "$"*(end-start)   + text[end:]
            return [item] + self._extract(new_text, re_patterns)
        else:
            return []
        
    def melt_entities(self, items,label2entities):
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

        patterns = self._build_regexp_patterns(label2entities)
        items = self._extract(text, patterns)
        # 拆分多个实体为列表
        items = self.melt_entities(items, label2entities)
        # 如果没有提取到贴现人（贴现人在最后）
        discounter_items = list(filter(lambda x:"贴现人" in x, items))
        labels = sorted(labels, key=lambda x:x["start"])   # 按索引位置排序
        # discounter_labels = filter(lambda x:x["label_name"]=="贴现人",labels[-2:])   # 最后两个标签
        discounter_labels = filter(lambda x:x["label_name"]=="贴现人",labels)   # 最后两个标签
        discounters = list(map(lambda x:x["text"],discounter_labels))
        if len(discounter_items)==0 and len(discounters) > 0:
            for item in items:
                item["贴现人"] = discounters
        # 如果没有提取到票据期限（票据期限在第一个位置)
        due_items = list(filter(lambda x:"票据期限" in x, items))
        first_label = labels[0]
        if first_label["label_name"] == "票据期限" and len(due_items) == 0:
            for item in items:
                item["票据期限"] = first_label["text"]
        return items


class TemplateExtractor(Extractor):
    """ 模板抽取器
    """
    def __init__(self, templates:list=None):
        if isinstance(templates,list):
            self.templates = templates
        else:
            self.templates = []

    def add_template(self, template):
        self.templates.append(template)
    
    def add_templates(self, templates:list):
        self.templates.extend(templates)

    def _build_regexp_patterns(self, tags):
        label2express = dict()
        for k, v in tags.items():
            if k == "贴现人":
                v = [vv+"贴?" for vv in v]
                val = join_entities_as_regexp(v, prefix="/?")
            elif k == "承兑人":
                val = join_entities_as_regexp(v, prefix="[、,，\s及和/+或者]{0,3}",sufix="(?!贴)")
            elif k == "票据期限":
                val = join_entities_as_regexp(v, prefix="",sufix="[\.\s、，及和至或者\-+/]*")
            else:
                val = join_entities_as_regexp(v)
            label2express[k] = val

        base_mapping = {label:"(?P<{name}>(?:{exp})+)".format(name=label,exp=exp) for label, exp in label2express.items()}
            
        regexps = []
        for template in self.templates:
            mapping = dict(base_mapping)
            # 若模板中的标签名存在数字后缀，则补充带数字后缀的命名分组表达式（正则）
            nums = re.findall("(?<!\{)\{.*?(\d+)\}(?!\})", template)
            if len(nums) > 0:
                max_num = max(map(int, nums))
                for label, express in label2express.items():
                    for i in range(1,max_num+1):
                        new_label = label+str(i)
                        mapping[new_label] = "(?P<{name}>{exp})".format(name=new_label,exp=express)

            # 填充模板
            try:
                regexp = template.format_map(mapping)
            except:
                # warnings.warn("template `%s` fill element %s failed"%(r, mapping))
                continue
            else:
                regexps.append(regexp)

        patterns = []
        for regexp in regexps:
            try:
                p = re.compile(regexp)
            except:
                warnings.warn("regexp `%s` compile failed"%regexp)
                continue
            else:
                patterns.append(p)
        return patterns
    
    def _extract_and_assemble_info(self, p, text):
        # 1. 正则抽取
        m = p.search(text)
        if m:
            rs_ = m.groupdict()
            rs = {k:v for k,v in rs_.items() if v is not None}
        else:
            rs = dict()

        # 2. 组装
        num_p = re.compile("\d+$")
        common_item = dict()
        tmp = defaultdict(dict)
        ## 2.1 分离标签的后缀数字
        for k, v in rs.items():
            nums = num_p.findall(k)
            if len(nums) == 0:
                common_item[k]=v
            else:
                num = nums[0]
                kk = num_p.sub("",k)
                tmp[num][kk] = v
        output = []
        ## 2.2 填充共有标签实体
        if len(tmp) > 0:
            for k, d in tmp.items():
                d.update(common_item)
                output.append(d)
        else:
            if len(common_item) > 0:
                output.append(common_item)
        return output

    def extract(self, text, labels):
        label2entities = defaultdict(set)
        for info in labels:
            label_name = info["label_name"]
            entity = info["text"]
            entity = entity.replace("\\",'\\\\')
            label2entities[label_name].add(entity)      

        patterns = self._build_regexp_patterns(label2entities)
        data = [self._extract_and_assemble_info(p, text) for p in patterns]
        if len(data) > 0:
            items = max(data, key=lambda x:sum([len(xx) for xx in x]))
        else:
            items = []
        return items


class IntegrateExtractor(Extractor):
    """组合关系抽取器，把出现的标签直接组合成一组抽取信息"""
    def __init__(self, multival_label=None):
        self.multival_label = multival_label

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


           