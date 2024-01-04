import re
from collections import  Counter, defaultdict

from . import split as Spliter


def is_unique(alist):
    return len(alist) == len(set(alist))



def join_entity(ents):
    join_regexp = "[、,，\s及和/+或者至]{0,3}"
    _ents = [join_regexp + ent for ent in ents]
    regexp = "|".join(_ents)
    return regexp


        

class ReExtractor:
    def __init__(self):
        self.rules = list()
        self.spliter = Spliter

    def add_rule(self, rule):
        self.rules.append(rule)

    def _build_regexp(self, tags):
        patterns = []
        mapping = dict()
        for k, v in tags.items():
            if k in ("承兑人","贴现人","票据期限"):
                # v = ["[、,，\s及和/+或者至]{0,3}" + vv for vv in v]
                # val =  "|".join(v)
                val = join_entity(v)
                # val = "(%s)+"%val
                val = "((?:%s)+)"%val
            else:
                val =  "|".join(v)
            mapping[k] = "(?P<{name}>{exp})".format(name=k,exp=val)
            
        for r in self.rules:
            try:
                rexp = r.format_map(mapping)
            except:
                continue 
            else:
                try:
                    p = re.compile(rexp)
                except:
                    print("complie error:",rexp,tags)
                    continue
                else:
                    patterns.append(p)
        return patterns
    
    def _extract(self, p, text):
        m = p.search(text)
        if m:
            rs = m.groupdict()
            rs = {k:v for k,v in rs.items() if v is not None}
            rs["matched_pattern"] = p.pattern
            return rs
        else:
            return dict()


    def extract(self, text, labels):
        category = defaultdict(set)
        for info in labels:
            label_type = info["label_name"]
            label_text = info["text"]
            category[label_type].add(label_text)        

        regexps = self._build_regexp(category)
        data = [self._extract(r, text) for r in regexps]
        if len(data) > 0:
            rst = max(data, key=lambda x:len(x))
        else:
            rst = dict()
        return rst