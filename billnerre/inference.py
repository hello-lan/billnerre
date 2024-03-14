import torch as t

from utils.mix import get_entities

# 命名实体识别（要素抽取）
class NerWrapper:
    def __init__(self, model,vocab, id2label):
        self.model = model
        self.vocab = vocab
        self.id2label = id2label
        self.init()

    def init(self):
        self.model.eval()

    def prepare_input(self, text):
        words = list(text)
        input_ids = [self.vocab.to_index(w) for w in words]
        input_mask = [True] * len(words)
        input_ids = t.tensor([input_ids], dtype=t.long)
        input_mask = t.tensor([input_mask], dtype=t.bool)
        return input_ids, input_mask    
    
    def predict(self, text):
        input_ids, input_mask = self.prepare_input(text)
        with t.no_grad():
            tagids = self.model.forward_tags(input_ids, input_mask)
        label_entities = get_entities(tagids[0], self.id2label)
        items = []
        for label_name, start, end in label_entities:
            words = text[start:end+1]
            item = dict(label_name=label_name,start=start,end=end,text=words)
            items.append(item) 
        tags = [self.id2label[tag_id] for tag_id in tagids[0]] 
        return items, tags
    
    @classmethod
    def load_model(cls, config):
        from models.bilstm_crf import BiLSTM_CRF
        from utils import Vocabulary
        ner_model, word2id = BiLSTM_CRF.load_model(config.model_path)
        vocab = Vocabulary(word2id)
        return cls(ner_model,vocab,config.id2label)


def init_model(config):
    global modelwrapper
    modelwrapper = NerWrapper.load_model(config)


def predict(text):
    return modelwrapper.predict(text)


# 关系抽取
def init_manager():
    global manager
    from relation.manager import RelationExtractorManager
    manager = RelationExtractorManager.of_default()


def extract_relation(text, items):
    extracts = manager.extract_core_relation(text,items)
    return extracts


if __name__ == "__main__":
    from config import Config
    init_model(Config)
    init_manager()

    text = "爱猴主人上海南京 章琳洁 新 : 收3月电商，国股大商贴"
    items, tags = predict(text)
    result = extract_relation(text, items)
    print(result)
