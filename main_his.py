import os

import click
import torch as t
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.baseline import BiLSTM_CRF
from config import Config
from utils.file_io import load_json
from utils.metric import SeqEntityScore
from utils.logger import logger, init_logger
from utils import Vocabulary, DatasetLoader, ProgressBar, AverageMeter


def get_or_build_vocab(conf):
    vocab = Vocabulary()
    vocab_path = conf.vocab_path
    if os.path.exists(vocab_path):
        vocab.load_from_file(vocab_path)
    else:
        fpaths = [conf.train_data_path, conf.eval_data_path]
        for fpath in fpaths:
            data = load_json(fpath)
            for item in data:
                text = item['text']
                vocab.update(list(text))
        vocab.build_vocab()
        vocab.save(vocab_path)
    return vocab


def train_model(model, vocab, conf):
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
            features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pbar(step=step, info={'loss': loss.item()})
            train_loss.update(loss.item(), n=1)
        train_log = {'loss': train_loss.avg}
        eval_log, class_info = evaluate_model(model, vocab, conf)
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
                info = f"Subject: {key} - Acc: {value['acc']} - Recall: {value['recall']} - F1: {value['f1']}"
                logger.info(info)


def evaluate_model(model, vocab, conf):
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
            features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
            eval_loss.update(val=loss.item(), n=input_ids.size(0))
            # tags, _ = model.crf._obtain_labels(features, conf.id2label, input_lens)
            tags = model.forward_tags(input_ids, input_mask, conf.id2label,input_lens)
            input_tags = input_tags.cpu().numpy()
            target = [input_[:len_] for input_, len_ in zip(input_tags, input_lens)]
            metric.update(pred_paths=tags, label_paths=target)
            pbar(step=step)

    eval_info, class_info = metric.result()
    eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
    result = {'eval_loss': eval_loss.avg}
    result = dict(result, **eval_info)
    return result, class_info



@click.command()
@click.option('--train', is_flag=True, help='模型训练')
@click.option('--eval', is_flag=True, help="模型验证")
@click.option('-m', "--model", type=click.Choice(['bilstm_crf']), default='bilstm_crf', help="模型", show_default=True)
@click.option('--gpu', type=int, default=None, help='GPU')
def main(train,eval, model, gpu):
    if isinstance(gpu, int):
        device = t.device(f"cuda:{gpu}")
    else:
        device = t.device('cpu')
    conf = Config()
    vocab = get_or_build_vocab(conf)

    ner_model = BiLSTM_CRF(vocab_size=len(vocab),
                       embedding_size=conf.embeding_size,
                       hidden_size=conf.hidden_size,
                       device=device,
                       label2id=conf.label2id
                       ) 
    ner_model.to(device)

    if train:
        log_path = os.path.join(conf.cache_dir, f'{model}-train.log')
        init_logger(log_file=log_path)
        train_model(ner_model, vocab, conf)

    if eval:
        pass 



if __name__ == "__main__":
    main()