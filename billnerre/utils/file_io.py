import pickle
import json
from pathlib import Path


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


def load_pickle(input_file):
    '''
    读取pickle文件
    :param pickle_path:
    :param file_name:
    :return:
    '''
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path,mode='w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

