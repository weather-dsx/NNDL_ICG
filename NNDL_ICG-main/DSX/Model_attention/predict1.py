import os
import torch
from PIL import Image
from torchvision import transforms
import json
from model2 import TransformerCaptioningModel  # 假设你已经定义了这个模型
from configuartions import Config  # 假设你有一个配置类
import torch.nn.functional as F


# 加载模型
def load_model(model_path, vocab, config):
    model = TransformerCaptioningModel(
        image_code_dim=config.image_code_dim,
        vocab=vocab,
        vocab_size=len(vocab),
        word_dim=config.word_dim,
        num_heads=config.num_heads,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        device=config.device
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(config.device)
    return model


# 处理图片
def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet标准化
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 增加一个批次维度
    return image_tensor


# 生成描述
def generate_caption(model, image_tensor, vocab, config, max_length=20, beam_size=3):
    # 使用generate方法进行描述生成
    output = model.generate(image_tensor.to(config.device), memory=None, max_length=max_length, beam_size=beam_size)

    # 将词索引转换为单词
    idx_to_word = {idx: word for word, idx in vocab.items()}
    captions = []
    for seq in output:
        caption = [idx_to_word.get(idx, '<unk>') for idx in seq]
        caption_str = ' '.join(caption).replace('<start>', '').replace('<end>', '').strip()
        captions.append(caption_str)

    return captions[0]  # 返回第一个生成的描述


# 加载配置和词汇表
config = Config()
vocab_path = '../data/output/vocab.json'

with open(vocab_path, 'r') as f:
    vocab = json.load(f)

# 加载模型
model_path = '../data/output/weights/Transformer_model_10257.3730.pth'
model = load_model(model_path, vocab, config)

# 图片文件夹路径
images_folder = "../data/images"  # 你需要提供实际的图片文件夹路径
captions_dict = {}

count = 1

# 遍历图片文件夹
for filename in os.listdir(images_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(images_folder, filename)

        # 处理图片并生成描述
        image_tensor = process_image(img_path)
        generated_caption = generate_caption(model, image_tensor, vocab, config)

        print(f"No{count}: {generated_caption}")
        count += 1

        # 将描述保存到字典
        captions_dict[img_path] = generated_caption

# 保存字典到JSON文件
output_path = "../data/output/captions.json"  # 保存路径
with open(output_path, 'w') as json_file:
    json.dump(captions_dict, json_file, indent=4)

print(f"Captions saved to {output_path}")
