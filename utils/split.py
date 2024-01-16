import re


def _check_split(labes, idx):
    pass


def split_text_labels(text, labels, split_idx):
    _check_split(labels, split_idx)
    left_text, right_text = text[:split_idx], text[split_idx:]
    left_labels, right_labels = [], []
    for label_info in labels:
        info = dict(label_info)
        if info["end"] < split_idx:
            left_labels.append(info)
        else:
            info['start'] -= split_idx
            info['end'] -= split_idx
            right_labels.append(info)
    return left_text, left_labels, right_text, right_labels


def split_publisher_vs_msg(text, labels):
    """ 分段: 发布者信息 与 消息内容"""
    i = text.find(":")
    publisher_text, publisher_labels, msg_text, msg_labels = split_text_labels(text, labels, i+2)
    return publisher_text, publisher_labels, msg_text, msg_labels
    

def split_msg(text, labels):
    """ 分段：消息内容分段 """
    # m = re.search("[\r\n]+", text)      # 按换行符分割
    # m = re.search("([\r\n]+|[；，。\s][另再]?(?=出|收))", text)      # 按换行符及交易方向分割
    m = re.search("([\r\n]+|[；，。\s][另再]?(?=出|收)|.{3,}[,，。、；\s](?=[^,，。、；\s]{0,5}量?[出收买卖]断?(点|少量|：|\d+)))", text)      # 可能会出现金额收xxx
    if m:
        idx = m.span()[-1]
        left_text, left_labels, right_text, right_labels = split_text_labels(text, labels, idx)
        return split_msg(left_text.rstrip(), left_labels) + split_msg(right_text.rstrip(), right_labels)
    else:
        return [dict(text=text, labels=labels)]

