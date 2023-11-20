import json 


def preprocess(path):
    data = []
    with open(path,'r') as f:
        for line in f:
            item = json.loads(line)
            text = item['text']
            label_ent = item.get('label',None)
            words = list(text)
            labels = ['O'] * len(words)
            if label_ent is not None:
                for key, value in label_ent.items():
                    for sub_name, sub_index in value.items():
                        for start_index, end_index in sub_index:
                            assert ''.join(words[start_index:end_index + 1]) == sub_name
                            if start_index == end_index:
                                labels[start_index] = 'S-' + key
                            else:
                                labels[start_index] = 'B-' + key
                                labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
            info = dict(text=text, tag=" ".join(labels))
            data.append(info)
    return data



if __name__ == "__main__":
    path_train = 'cluener/train.json'
    path_test = 'cluener/dev.json'

    data_train = preprocess(path_train)
    data_test = preprocess(path_test)

    with open('../data/corpus/train_processed.json','w',encoding='utf-8') as f:
        json.dump(data_train, f, indent=2)

    with open('../data/corpus/dev_processed.json','w',encoding='utf-8') as f:
        json.dump(data_test, f, indent=2)