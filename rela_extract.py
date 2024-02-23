from relation import RelationExtractorManager
from utils.file_io import load_json, save_json


def load_test_data():
    path = "../corpus/Step2_已标注语料/final/WJQ_0118_final_2799.json"
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
    manager = RelationExtractorManager.of_default()
    # 加载数据
    data = load_test_data()
    tmp, useless = [],[]
    for i,item in enumerate(data):
        text = item["text"]
        labels = item["labels"]
        rst, noneuse = manager.extract_relation(text, labels)
        tmp.extend(rst)    
        useless.extend(noneuse)

    save_json(useless, "cache/useless_%s.json"%len(useless))
    
    def pop_labels(item):
        item_new = dict(item)
        item_new.pop("labels")
        return item_new
    
    def pop_matched_info(item):
        item_new = dict(item)
        for it in item_new["output"]:
            it.pop("matched_info")
        return item_new
    


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
        return len(y)  > 0 and item["method"]==7
    
    tmp_08 = list(map(pop_labels,filter(is_complete, tmp)))
    save_json(tmp_08,"cache/no_complete_%s.json"%len(tmp_08))

    tmp_subj = list(map(pop_labels,filter(lambda item:item["ext"] == "MultiSubjectExtractor(templates=[...])", tmp)))
    save_json(tmp_subj,"cache/subj_%s.json"%len(tmp_subj))

    ## flatten data
    new_tmp = []
    for item in tmp:
        for rela_item in item["output"]:
            new_item = dict(item)
            new_item["output"] = rela_item 
            new_tmp.append(new_item)
    for ext in manager.rela_extractors:
        # rst_ext = list(filter(lambda item: item["ext"]==str(ext), new_tmp))
        rst_ext = list(map(pop_labels,filter(lambda item: item["ext"]==str(ext), new_tmp)))
        save_json(rst_ext,"cache/%s_%s.json"%(str(ext),len(rst_ext)))



if __name__ == "__main__":
    test()