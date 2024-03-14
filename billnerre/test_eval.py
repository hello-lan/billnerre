from utils.metric import SeqEntityScore
from utils.file_io import load_json
from config import Config as conf
from pprint import pprint



if __name__ == "__main__":
    data = load_json("cache/predict_from_java.json")
    metric = SeqEntityScore(conf.id2label, markup=conf.markup)
    for item in data:
        labels = item["labels"]
        preds = item["preds"]
        metric.update([labels],[preds])

    score,class_info = metric.result()
    print(score)
    pprint(class_info)
