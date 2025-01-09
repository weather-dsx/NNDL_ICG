import torch
import json
import re

def load_data_tvt(dataname='train', word_to_idx=[]):
    data = None
    if dataname == 'train':
        with open('data/train.json', 'r') as json_file:
            data = json.load(json_file)
    elif dataname == 'valid':
        with open('data/valid.json', 'r') as json_file:
            data = json.load(json_file)
    elif dataname == 'test':
        with open('data/test_captions.json', 'r') as json_file:
            data = json.load(json_file)
    
    # 提取描述文本
    descriptions = list(data.values())
    
    # 分词并去除标点符号
    all_words = []
    for description in descriptions:
        words = re.findall(r'\b\w+\b|[,.-]', description)
        all_words.extend(words)
    
    # 编码为索引
    encoded_list = []
    for description in descriptions:
        words = description.lower().split()
        words = ['<sta>'] + words
        words.append('<eos>')
        encoded = torch.stack([torch.tensor(word_to_idx[word]) for word in words if word in word_to_idx], dim=0)
        encoded_list.append(encoded)
    
    # 构建 key-encoded 映射字典
    key_dict = dict(zip(data.keys(), encoded_list))
    
    # # 返回第一个句子的结果
    # first_sentence_key = list(key_dict.keys())[0]
    # first_sentence_encoded = key_dict[first_sentence_key]
    
    return key_dict