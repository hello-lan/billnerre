import pickle
import os
from pathlib import Path
import numpy as np


def save_pickle(data, file_path):
    '''
    保存成pickle文件
    :param data:
    :param file_name:
    :param pickle_path:
    :return:
    '''
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def read_pretrainded_file(path):
    vectors = []
    char2id = dict()
    with open(path) as f:
        for i,line in enumerate(f):
            if i == 0:
                continue
            infos = line.strip().split(" ")
            char = infos[0]
            vec = [float(x) for x in infos[1:]]
            if len(infos[0]) > 1:
                continue
            if char in char2id:
                print(line[:10])
                continue
            vectors.append(vec)
            char2id[char] = len(vectors) - 1
    return vectors, char2id


if __name__ == "__main__":
    # path = 'data/pre_trained_embedding/sgns.weibo.char'
    # save_path = "../billnerre/data/pretrained_embedding/sgns_weibo"

    path = 'data/pre_trained_embedding/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'
    save_path = "../billnerre/data/pretrained_embedding/tencent"

    vectors, char2id = read_pretrainded_file(path)
    # 向量平均值作为未知字符的向量
    vectors.append(np.array(vectors).mean(axis=0))
    char2id['[UNK]'] = len(vectors) - 1
    vectors = np.array(vectors)
    id2char = {v:k for k,v in char2id.items()}

    mappings = {
            "word2idx": char2id,
            'idx2word': id2char
        }
    
    vocab_path = os.path.join(save_path, 'vocab.pkl')
    emb_path = os.path.join(save_path, 'embedding.npy')

    save_pickle(mappings, vocab_path)
    np.save(emb_path,vectors,allow_pickle=True)
    
    print(vectors.shape)



