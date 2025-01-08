import os
import torch
from PIL import Image
from torchvision import transforms
from models import AttentionModel
from configurations import Config
import json

def load_model(model_path, vocab, config):
    model = AttentionModel(
        image_code_dim=config.image_code_dim,
        vocab=vocab,  # 传递词汇表字典
        word_dim=config.word_dim,
        attention_dim=config.attention_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(config.device)
    model.eval()
    return model

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加一个批次维度
    return image_tensor

def predict_caption(model, image_tensor, vocab, config):
    # 生成束搜索描述
    predictions = model.generate_by_beamsearch(image_tensor.to(config.device), config.beam_k, config.max_len)
    # 将词索引转换回文字
    idx_to_word = {idx: word for word, idx in vocab.items()}
    caption_words = [idx_to_word.get(word, '<unk>') for word in predictions[0]]
    caption = ' '.join(caption_words)
    return caption

def load_true_captions(true_captions_path):
    """加载真实描述数据"""
    with open(true_captions_path, 'r') as file:
        true_captions = json.load(file)
    return true_captions

def main():
    # 配置和词汇表加载
    config = Config()
    vocab_path = '../data/output/vocab.json'
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    # 加载模型
    model_path = '../data/output/weights_64/Attention_model_background_caption_1.2696454833215913.pth'
    model = load_model(model_path, vocab, config)
    
    # 真实描述文件路径
    true_captions_path = '../data/data_captions.json'
    true_captions = load_true_captions(true_captions_path)
    
    # 图片文件夹路径
    images_folder = "../data/images"
    captions_dict = {}

    count = 1

    # 遍历图片文件夹，处理每张图片
    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(images_folder, filename)

            # 获取真实描述
            true_caption = true_captions.get(filename, 'No caption available')  # 如果找不到描述，默认返回'No caption available'

            # 加载和处理图片
            image_tensor = process_image(img_path)

            # 生成描述
            generated_caption = predict_caption(model, image_tensor, vocab, config)

            # 输出
            print(f"No{count}_generate: {generated_caption}")
            #print(f"No{count}_true: {true_caption}")
            count += 1

            # 保存到字典，包含真实描述和生成描述
            captions_dict[filename] = {
                "true_caption": true_caption,
                "generated_caption": generated_caption
            }

    # 保存字典到JSON文件
    output_path = "../data/output/captions_true_generated_64_1.26_attention.json"
    with open(output_path, 'w') as json_file:
        json.dump(captions_dict, json_file, indent=4)

    print(f"Captions saved to {output_path}")

if __name__ == '__main__':
    main()
