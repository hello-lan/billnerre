import os

import click
import numpy as np
import torch as t
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.bilstm_crf import BiLSTM_CRF
from config import Config
from utils.file_io import load_json,save_json
from utils.metric import SeqEntityScore
from utils.logger import logger, init_logger
from utils import VocabularyBuilder,Vocabulary,load_pretrained_vocab_embedding, DatasetLoader, ProgressBar, AverageMeter
from utils.mix import get_entities


def train(model, vocab, conf):
    train_data = load_json(conf.train_data_path)
    train_dataloader = DatasetLoader(data=train_data, 
                                     batch_size=conf.batch_size,
                                     shuffle=False,
                                     seed=conf.seed,
                                     sort=True,
                                     vocab=vocab,
                                     label2id=conf.label2id
                                     )
    
    optimizer = model.get_optimizer(lr=conf.lr,weight_decay=conf.lr_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                  verbose=1, cooldown=0, min_lr=0, eps=1e-8)
    best_f1 = 0
    for epoch in range(1, conf.epochs+1):
        print(f"Epoch {epoch}/{conf.epochs}")
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        train_loss = AverageMeter()
        model.train()
        assert model.training
        for step, batch in enumerate(train_dataloader):
            input_ids, input_mask, input_tags, input_lens = batch
            input_mask = input_mask > 0
            loss = model.forward_loss(input_ids, input_mask, input_tags)
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pbar(step=step, info={'loss': loss.item()})
            train_loss.update(loss.item(), n=1)
        train_log = {'loss': train_loss.avg}
        eval_log, class_info = evaluate(model, vocab, conf)
        logs = dict(train_log, **eval_log)
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        logger.info(show_info)
        scheduler.step(logs['eval_f1'], epoch)
        if logs['eval_f1'] > best_f1:
            logger.info(f"\nEpoch {epoch}: eval_f1 improved from {best_f1} to {logs['eval_f1']}")
            logger.info("save model to disk.")
            best_f1 = logs['eval_f1']
            # model.save()
            model.save(conf.model_path, vocab.word2idx)

            print("Eval Entity Score: ")
            for key, value in class_info.items():
                info = f"Subject: {key} - Precision: {value['precision']} - Recall: {value['recall']} - F1: {value['f1']}"
                logger.info(info)


def evaluate(model, vocab, conf):
    eval_data = load_json(conf.eval_data_path)
    eval_dataloader = DatasetLoader(data=eval_data, 
                                    batch_size=conf.batch_size,
                                    shuffle=False,
                                    seed=conf.seed,
                                    sort=False,
                                    vocab=vocab,
                                    label2id=conf.label2id
                                    )   

    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    eval_loss = AverageMeter()
    metric = SeqEntityScore(conf.id2label, markup=conf.markup)
    model.eval()
    with t.no_grad():
        for step, batch in enumerate(eval_dataloader):
            input_ids, input_mask, input_tags, input_lens = batch
            input_mask = input_mask > 0
            loss = model.forward_loss(input_ids, input_mask, input_tags)
            eval_loss.update(val=loss.item(), n=input_ids.size(0))
            tags = model.forward_tags(input_ids, input_mask)
            input_tags = input_tags.cpu().numpy()
            target = [input_[:len_] for input_, len_ in zip(input_tags, input_lens)]
            metric.update(pred_paths=tags, label_paths=target)
            pbar(step=step)

    eval_info, class_info = metric.result()
    eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
    result = {'eval_loss': eval_loss.avg}
    result = dict(result, **eval_info)
    return result, class_info


def predict(model,vocab,conf):
    lines = [
        # "邮储黄C+15386699183 : 出4月到期乐山，邮储直贴",
        # "浦发天津-胡广森 : 询价收12月电\n浦发天津-胡广森18221951727",
        # "同创合鑫 赵达超15215707830 : 收跨年重庆 青岛农 河北 成都 天津 九江 江西 郑州 中原 稠州 宁波通商杭州联合农 萧山农 江南农 等优质城农 城贴以上可自贴\n收1月遂宁 或者2月初到期 非村贴",
        # "吾见青山 重庆富民 喻飞 : 1、出城商\n2、接票据过桥，授信广，速度快！\n3、富民银行（AA+）收各期限拆借，限量发行各期限存单！",
        "吾见青山 重庆富民 喻飞 : 1、出城商",
        # "浦发天津-胡广森 : 1、出城商",
        # "诗情悦意 : \n【收】2月3月4月到期授信城农商、财司、商票，国股大商贴\n闫诗悦邮储黑龙江13199464913",
        # "同创合鑫 赵达超15215707830 : 收北部湾 大连 秦农 城贴/农贴",
    ]
    for line in lines:
        words = list(line)
        input_ids = [vocab.to_index(w) for w in words]
        input_mask = [True] * len(words)
        model.eval()
        with t.no_grad():
            input_ids = t.tensor([input_ids], dtype=t.long)
            input_mask = t.tensor([input_mask], dtype=t.bool)
            tags = model.forward_tags(input_ids, input_mask)
        # print(tags)
        label_entities = get_entities(tags[0], conf.id2label)
        items = []
        for label_name, start, end in label_entities:
            words = line[start:end+1]
            item = dict(label_name=label_name,start=start,end=end,text=words)
            items.append(item)
        rst = dict(text=line, labels=items)
        print(rst)
        print("\n") 


def extract(model, vocab, conf):
    from relation.manager import RelationExtractorManager
    rela_ext_manager = RelationExtractorManager.of_default()

    data = load_json(conf.eval_data_path)
    model.eval()
    results = []
    for item in data:
        text = item["text"]
        words = list(text)
        input_ids = [vocab.to_index(w) for w in words]
        input_mask = [True] * len(words)

        with t.no_grad():
            input_ids = t.tensor([input_ids], dtype=t.long)
            input_mask = t.tensor([input_mask], dtype=t.bool)
            tags = model.forward_tags(input_ids, input_mask)
        label_entities = get_entities(tags[0], conf.id2label)
        items = []
        for label_name, start, end in label_entities:
            words = text[start:end+1]
            item = dict(label_name=label_name,start=start,end=end,text=words)
            items.append(item)
        # rst = dict(text=text, labels=items)
        # 关系抽取
        relation = rela_ext_manager.extract_relation(text,items)
        relation.pop("labels")
        relation.pop("empties")
        results.append(relation)
    
    save_path = "cache/ner_relation_result.json"
    save_json(results, save_path)


@click.command()
@click.option('-t', '--task', type=click.Choice(['train', 'eval','predict','extract']),default='train', help='任务')
@click.option("--revocab", is_flag=True, help="是否重新创建vocablary")
@click.option('--pretrained', is_flag=True, help='使用预训练词向量')
@click.option('--gpu', type=int, default=None, help='GPU')
def main(task, gpu, pretrained,revocab):
    if isinstance(gpu, int):
        device = t.device(f"cuda:{gpu}")
    else:
        device = t.device('cpu')
    conf = Config()

    if task == "train":
        # 是否使用预训练词向量
        if pretrained:
            click.echo("读取预训练词向量...")
            # pretrained_dir = "data/pretrained_embedding/sgns_weibo"
            pretrained_dir = "data/pretrained_embedding/tencent"
            vocab, embedding = load_pretrained_vocab_embedding(pretrained_dir)
            conf.embeding_size= embedding.shape[1]
        else:
            click.echo("加载或创建vocabulary...")
            vocab = VocabularyBuilder.get_or_build_vocab(conf,rebuild=revocab)

        ner_model = BiLSTM_CRF(vocab_size=len(vocab),
                               embedding_size=conf.embeding_size,
                               hidden_size=conf.hidden_size,
                               label_size=len(conf.label2id)
                               )
    
        if pretrained:
            # 加载预训练词向量
            click.echo("加载预训练词向量到模型...")
            ner_model.load_pretrained_embedding(embedding)

        log_path = os.path.join(conf.cache_dir, 'train.log')
        init_logger(log_file=log_path)
        # ner_model.to(device)
        click.echo("模型训练...")
        train(ner_model, vocab, conf)

    else:
        ner_model, word2id = BiLSTM_CRF.load_model(conf.model_path)
        vocab = Vocabulary(word2id)

        if task == "predict":
            predict(ner_model, vocab, conf)
    
        if task == "extract":
            extract(ner_model, vocab, conf)
        
        if task == "eval":
            from pprint import pprint
            rst, classinfo = evaluate(ner_model, vocab, conf)
            pprint(rst)
            pprint(classinfo)


if __name__ == "__main__":
    main()
