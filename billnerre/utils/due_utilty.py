import re
from collections import namedtuple


DueItem = namedtuple("dueItem",["month","due"],defaults=[None,None])


def get_digit_month(txt):
    m = re.search("(1[012]|\d)\s*月", txt)
    if m:
        return DueItem(month=int(m.group(1)))
    else:
        return None


_cnns = ["一","二","三","四","五","六","七","八","九","十","十一","十二"]  # 注意顺序
cnn2digit = {x:i for i,x in enumerate(_cnns,1)}

def get_cn_num_month(txt):
    m = re.search("(十[一二]|[一二三四五六七八九十])\s*月",txt)
    if m:
        cn_num = m.group(1)
        return DueItem(month=cnn2digit.get(cn_num))
    else:
        return None


q2m = {"一":3,"二":6,"三":9,"四":12,"1":3,"2":6,"3":9,"4":12}

def get_quarter_month(txt):
    m = re.search("([一二三四1234])季度",txt)
    if m:
        q = m.group(1)
        return DueItem(month=q2m.get(q))
    else:
        return None
    
def get_curr_month(txt):
    m = re.search("(当月|月[初中末底内]|本月|\d+[日号]|上旬|下旬|中旬|托收)",txt)
    if m:
        return DueItem(due=0)
    else:
        return None
    
def get_md_month(txt):
    months = re.findall("(1[012]|\d)\.[0123]?\d",txt)
    if len(months) > 0:
        m = max(map(int,months))
        return DueItem(month=m)
    else:
        return None

def get_due(txt):
    m = re.search("(\d+)M",txt)
    if m:
        return DueItem(due=int(m.group(1)))
    else:
        return None
    

def get_january(txt):
    if re.search("跨年\s*$",txt):
        return DueItem(month=1)
    else:
        return None

       
month_getters = [get_digit_month,
                 get_cn_num_month,
                 get_quarter_month,
                 get_due,
                 get_md_month,
                 get_curr_month,
                 get_january,
                ]

def extract_dueItem_from_duetext(txt):
    if not isinstance(txt,str):
        return DueItem()
    for month_getter in month_getters:
        item = month_getter(txt)
        if item is not None:
            return item
    else:
        return DueItem()
    
def normlize_due(txts, cur):
    """ 
    txts: list[str], 模型抽取的原始`票据期限`实体
    cur: int, 当前月份
    """
    items = map(extract_dueItem_from_duetext,txts)
    months, dues = zip(*items)
    due_1 = max(dues, key=lambda x:x if isinstance(x, (int,float)) else -float("inf"))
    if isinstance(due_1, (int,float)):
        due = "托收" if due_1 == 0 else "%dM"%due_1
        return [due]
    elif isinstance(cur, (int, float)):
        month = max(months, key=lambda x: -float("inf") if not isinstance(x, (int,float)) else x - cur  if  x > cur  else x - cur + 12)  # +12是因为月份跨年
        if isinstance(month, (int,float)):
            due_2 = month - cur if month > cur else month - cur + 12
            due = "托收" if due_2 == 0 else "%dM"%due_2
            return [due]
        else:
            return txts
    else:
        return txts
    