import re
import json
from collections import Counter

# 指定映射表文件路径
mapping_file = 'word_mapping.json'

# 读取 JSON 文件
with open('data/train_captions.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 提取描述部分
descriptions = []
for key, value in data.items():
    descriptions.append(value)

# 将所有描述合并成一个字符串
combined_text = " ".join(descriptions)

# 将逗号和句号替换为空格
combined_text = re.sub(r'[.,]', ' ', combined_text)

# 拆分成单词并去重
words = combined_text.split()
unique_words = sorted(set(words))
vocab = ['<pad>', '<sta>', '<eos>',',','.'] + list(unique_words)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for idx, word in enumerate(vocab)}

# 保存映射表到文件
with open(mapping_file, 'w', encoding='utf-8') as f:
    json.dump({'word_to_idx': word_to_idx, 'idx_to_word': idx_to_word, 'vocab': vocab}, f, ensure_ascii=False, indent=4)
print("映射表生成并保存成功。")