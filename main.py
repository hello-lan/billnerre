import os

import click
import numpy as np
import torch as t
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.bilstm_crf import BiLSTM_CRF
from config import Config
from utils.file_io import load_json
from utils.metric import SeqEntityScore
from utils.logger import logger, init_logger
from utils import get_or_build_vocab, load_pretrained_vocab_embedding, DatasetLoader, ProgressBar, AverageMeter
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
            model.save()

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
    # path = "checkpoints/BiLSTM_CRF_best_20231121.pth"
    # model.load(path)
    lines = ["四川敦煌学”。近年来，丹棱县等地一些不知名的石窟迎来了海内外的游客，他们随身携带着胡文和的著作。",
             "即使在收藏市场上已基本绝迹，但广州货币金融博物馆拥有全套。是2000年旧馆开馆时，",
             "原来提丽泽商务区开发好几年了，没有大的进展，原因是定位一直不好，所以定位的明确，而且坚持下去，"
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
        label_entities = get_entities(tags[0], conf.id2label)
        items = []
        for tag, start, end in label_entities:
            words = line[start:end+1]
            item = dict(tag=tag,start=start,end=end,entity=words)
            items.append(item)
        # print(line)
        print(items) 


@click.command()
@click.option('-t', '--task', type=click.Choice(['train', 'eval','predict']),default='train', help='任务')
@click.option("--revocab", is_flag=True, help="是否重新创建vocablary")
@click.option('--pretrained', is_flag=True, help='使用预训练词向量')
@click.option('-m', "--model", type=click.Choice(['bilstm_crf']), default='bilstm_crf', help="模型", show_default=True)
@click.option('--gpu', type=int, default=None, help='GPU')
def main(task, model, gpu, pretrained,revocab):
    if isinstance(gpu, int):
        device = t.device(f"cuda:{gpu}")
    else:
        device = t.device('cpu')
    conf = Config()

    # 是否使用预训练词向量
    if pretrained:
        click.echo("读取预训练词向量...")
        # pretrained_dir = "data/pretrained_embedding/sgns_weibo"
        pretrained_dir = "data/pretrained_embedding/tencent"
        vocab, embedding = load_pretrained_vocab_embedding(pretrained_dir)
        conf.embeding_size= embedding.shape[1]
    else:
        click.echo("加载或创建vocabulary...")
        vocab = get_or_build_vocab(conf,rebuild=revocab)

    if task == "train":
        ner_model = BiLSTM_CRF(vocab_size=len(vocab),
                               embedding_size=conf.embeding_size,
                               hidden_size=conf.hidden_size,
                               label_size=len(conf.label2id)
                               )
    
        if pretrained:
            # 加载预训练词向量
            click.echo("加载预训练词向量到模型...")
            ner_model.load_pretrained_embedding(embedding)


        log_path = os.path.join(conf.cache_dir, f'{model}-train.log')
        init_logger(log_file=log_path)
        # ner_model.to(device)
        click.echo("模型训练...")
        train(ner_model, vocab, conf)

    if task == "eval":
        pass
    if task == "predict":
        ner_model, _ = BiLSTM_CRF.load_model("checkpoints/BiLSTM_CRF_best_20231122.pth")
        predict(ner_model, vocab, conf)



if __name__ == "__main__":
    main()