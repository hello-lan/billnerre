labels = [
         'O',
         'B-address',
         'B-book',
         'B-company',
         'B-game',
         'B-government',
         'B-movie',
         'B-name',
         'B-organization',
         'B-position',
         'B-scene',
         'I-address',
         'I-book',
         'I-company',
         'I-game',
         'I-government',
         'I-movie',
         'I-name',
         'I-organization',
         'I-position',
         'I-scene',
         'S-address',
         'S-book',
         'S-company',
         'S-game',
         'S-government',
         'S-movie',
         'S-name',
         'S-organization',
         'S-position',
         'S-scene',
         '<START>',
         '<STOP>'
         ]


class Config:
    # 文件路径
    train_data_path = 'data/corpus_clue/train_processed.json'
    eval_data_path = 'data/corpus_clue/dev_processed.json'
    vocab_path = 'data/vocab_clue.pkl'
    cache_dir = 'cache'

    markup = 'bios'

    # 训练参数
    seed = 2023
    batch_size = 32
    lr = 0.001
    lr_decay = 0
    epochs = 5
    grad_norm = 5

    # 模型参数
    embeding_size = 128
    hidden_size = 384

    # 模型标签参数
    label2id = {label:i for i, label in enumerate(labels)}
    id2label = {i:label for i, label in enumerate(labels)}




