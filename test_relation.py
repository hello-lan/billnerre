from functools import reduce
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
    results = []
    for i,item in enumerate(data):
        text = item["text"]
        labels = item["labels"]
        result = manager.extract_relation(text, labels)
        results.append(result)

    save_json(results, "cache/test_relation_results_%s.json"%len(results))

    relations = reduce(lambda x,y:x+y, map(lambda x:x["relations"],results))
    save_json(relations, "cache/test_relation_relations_%s.json"%len(relations))

    empties = reduce(lambda x,y:x+y, map(lambda x:x["empties"],results))
    save_json(empties, "cache/test_relation_empties_%s.json"%len(empties))
    

    def pop_labels(item):
        item_new = dict(item)
        item_new.pop("labels")
        return item_new 

    def is_complete(item):
        labels = item["labels"]
        output = item["results"]
        entities_01 = [x["text"] for x in labels]
        entities_02_ = [xx for x in output for xx in x.values()]
        entities_02 = []
        for e in entities_02_:
            if isinstance(e, list):
                entities_02.extend(e)
            elif isinstance(e, str):
                entities_02.append(e)
        y = set(entities_01) - set(entities_02)
        return len(y)  > 0
    
    not_complete = list(map(pop_labels,filter(is_complete, relations)))
    save_json(not_complete,"cache/test_relation_not_complete_%s.json"%len(not_complete))

    ## flatten data
    for ext in manager.rela_extractors:
        # rst_ext = list(filter(lambda item: item["ext"]==str(ext), new_tmp))
        rst_ext = list(map(pop_labels,filter(lambda item: item["relaExtractor"]==str(ext), relations)))
        size = reduce(lambda x,y: x+y, map(lambda x:len(x["results"]),rst_ext))
        save_json(rst_ext,"cache/test_relation_%s_%s_total_%s.json"%(str(ext),len(rst_ext),size))


if __name__ == "__main__":
    test()