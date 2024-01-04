labels = [
         'O',
         'B-发布者所属机构',
         'I-发布者所属机构',
         'S-发布者所属机构',
         'B-票据期限',
         'I-票据期限',
         'S-票据期限',
         'B-承兑人',
         'I-承兑人',
         'S-承兑人',
         'B-贴现人',
         'I-贴现人',
         'S-贴现人',
         'B-利率',
         'I-利率',
         'S-利率',
         'B-金额',
         'I-金额',
         'S-金额',
        #  'B-票据种类',
        #  'I-票据种类',
        #  'S-票据种类',
        #  'B-承接业务',
        #  'I-承接业务',
        #  'S-承接业务',
         ]


class Config:
    # 文件路径
    train_data_path = 'data/corpus_msg/train_processed.json'
    eval_data_path = 'data/corpus_msg/dev_processed.json'
    vocab_path = 'data/vocab_msg.pkl'
    cache_dir = 'cache'

    markup = 'bios'

    # 训练参数
    seed = 2023
    batch_size = 15
    lr = 0.001
    lr_decay = 0
    epochs = 20
    grad_norm = 5

    # 模型参数
    embeding_size = 128
    hidden_size = 384

    # 模型标签参数
    label2id = {label:i for i, label in enumerate(labels)}
    id2label = {i:label for i, label in enumerate(labels)}




