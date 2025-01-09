import torch
import torch.nn as nn
import torchvision.models as models
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from dataset import load_data_tvt
from torchvision import transforms

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 数据预处理
mapping_file = 'word_mapping.json'

with open(mapping_file, 'r', encoding='utf-8') as f:
    mapping = json.load(f)
    word_to_idx = mapping['word_to_idx']
    idx_to_word = {int(k): v for k, v in mapping['idx_to_word'].items()}
    vocab = mapping['vocab']
    print("映射表加载成功。")
    
train_key_dict = load_data_tvt('train', word_to_idx)
valid_key_dict = load_data_tvt('valid', word_to_idx)
test_key_dict = load_data_tvt('test', word_to_idx)

class CustomDataset(Dataset):
    def __init__(self, key_dict, max_length=70, transform=None):
        self.key_dict = key_dict
        self.keys = list(key_dict.keys())
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        image = Image.open("data/images/" + key).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        encoded = self.key_dict[key]
        Length = len(encoded)
        pad_length = self.max_length - Length
        if pad_length > 0:
            onehot_encoded_max = torch.stack([torch.tensor(word_to_idx['<pad>']) for _ in range(pad_length)])
            caption = torch.cat([encoded, onehot_encoded_max])
        else:
            caption = encoded

        return image, caption

# transform转换操作组合，适配像 ResNet101 等深度学习模型的输入要求
transform2ResNet101 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# 对批次数据进行预处理，满足批量数据输入的格式要求
def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    padded_captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
    return images, padded_captions

# 超参数设置
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4

# 数据加载器
train_dataset = CustomDataset(train_key_dict, transform=transform2ResNet101)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)

val_dataset = CustomDataset(valid_key_dict, transform=transform2ResNet101)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

test_dataset = CustomDataset(test_key_dict, transform=transform2ResNet101)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)


# 模型结构
class CNN_GRU_Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super(CNN_GRU_Model, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256)  # 最终降维到256
        )
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(0.5)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, image_input, tgt):
        image_features = self.resnet(image_input)
        hidden_state = image_features.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)
        embedded = self.embedding(tgt)
        gru_output, _ = self.gru(embedded, hidden_state)
        output = self.fc(gru_output)
        return output
    
    # 预测函数
    @torch.no_grad()
    def predict(self, image_input, max_length=70):
        image_features = self.resnet(image_input)
        generated_sequences = []
        for feat in image_features:
            hidden_state = feat.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)
            input_sequence = torch.tensor([word_to_idx['<sta>']]).unsqueeze(0).to(image_input.device)
            seq = []
            for _ in range(max_length):
                embedded = self.embedding(input_sequence)
                gru_output, hidden_state = self.gru(embedded, hidden_state)
                predicted_token = self.fc(gru_output[:, -1, :])
                predicted_token = torch.argmax(predicted_token, dim=-1)
                seq.append(predicted_token.item())
                if predicted_token.item() == word_to_idx['<eos>']:
                    break
                input_sequence = predicted_token.unsqueeze(0)
            generated_sequences.append(seq)
        return generated_sequences
    
    # 描述生成函数
    def caption(self, image_path, max_length=70):
        # 加载并预处理图片
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform2ResNet101(image).unsqueeze(0).to(device)
        
        # 生成描述序列
        seq = self.predict(image_tensor, max_length=max_length)[0]
        
        # 将序列转换为文字
        caption = []
        for idx in seq:
            if idx == word_to_idx['<eos>']:
                break
            if idx != word_to_idx['<pad>'] and idx != word_to_idx['<sta>']:
                caption.append(idx_to_word[idx])
        
        return ' '.join(caption)

# 模型初始化
model = CNN_GRU_Model(
    vocab_size=len(vocab),
    embedding_size=256,  # 嵌入层维度降低到256
    hidden_size=256,     # GRU隐藏层维度降低到256
    num_layers=2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<pad>'])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)


# 评估函数
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

def evaluate(model, val_loader):
    model.eval()
    all_generated = []
    all_references = []
    with torch.no_grad():
        for images, captions in val_loader:
            images = images.to(device)
            captions = captions.to(device)
            generated_seq = model.predict(images)
            # 从生成的序列中去除 <pad> 和 <eos> 标记
            generated_caption_words = [[idx_to_word[idx] for idx in seq if idx not in [word_to_idx['<pad>'], word_to_idx['<eos>'], word_to_idx['<sta>']]] for seq in generated_seq]
            generated_caption_text = [' '.join(seq) for seq in generated_caption_words]
            # 从参考描述中去除 <pad> 和 <eos> 标记
            true_caption_words = [[idx_to_word[idx] for idx in cap.tolist() if idx not in [word_to_idx['<pad>'], word_to_idx['<eos>'], word_to_idx['<sta>']]] for cap in captions]
            true_caption_text = [' '.join(seq) for seq in true_caption_words]
            all_generated.extend(generated_caption_text)
            all_references.extend(true_caption_text)
    return all_generated, all_references

def calculate_metrics(all_generated, all_references):
    # 为 METEOR 分数标记化生成的描述和参考描述
    generated_tokens = [caption.split() for caption in all_generated]
    reference_tokens = [[caption.split()] for caption in all_references]
    meteor_scores = [meteor_score(ref, gen) for ref, gen in zip(reference_tokens, generated_tokens)]
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
    # 使用字符串计算 ROUGE-L 分数
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [scorer.score(reference, generated)['rougeL'].fmeasure for generated, reference in zip(all_generated, all_references)]
    avg_rouge_l_score = sum(rouge_l_scores) / len(rouge_l_scores)
    return avg_meteor_score, avg_rouge_l_score

from tqdm import tqdm

def train(model, train_loader, val_loader, optimizer, criterion, epochs=EPOCHS):
    model.train()
    best_meteor_score = -1
    best_rouge_l_score = -1
    
    for epoch in range(epochs):
        running_loss = 0.0
        # 使用 tqdm 显示进度
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for i, (images, captions) in enumerate(progress_bar):
            images = images.to(device)
            captions = captions.to(device)
            optimizer.zero_grad()
            output = model(images, captions[:, :-1])
            loss = criterion(output.view(-1, output.size(-1)), captions[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # 更新 tqdm 进度条的描述
            progress_bar.set_postfix(loss=running_loss / (i + 1))
        
        # 验证集评估
        all_generated, all_references = evaluate(model, val_loader)
        avg_meteor_score, avg_rouge_l_score = calculate_metrics(all_generated, all_references)
        print(f"Epoch [{epoch+1}/{epochs}] - METEOR: {avg_meteor_score:.4f}, ROUGE-L: {avg_rouge_l_score:.4f}")
        
        # 重新设置训练模式
        model.train()

        if (epoch + 1) % 1 == 0:
            if avg_meteor_score > best_meteor_score or avg_rouge_l_score > best_rouge_l_score:
                best_meteor_score = avg_meteor_score
                best_rouge_l_score = avg_rouge_l_score
                model_save_path = f"output1/epoch_{epoch+1}_meteor_{avg_meteor_score:.4f}_rougel_{avg_rouge_l_score:.4f}.pth"
                torch.save(model.state_dict(), model_save_path)
                print(f"Saved model at {model_save_path}")

import argparse
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

parser = argparse.ArgumentParser(description="Image Captioning Model")
parser.add_argument('--mode', type=int, choices=[1, 2], required=True, help="1 for training, 2 for generating")
args = parser.parse_args()

if args.mode == 1:
    # 训练模式
    train(model, train_loader, val_loader, optimizer, criterion, epochs=EPOCHS)
elif args.mode == 2:
    # 生成模式
    model.load_state_dict(torch.load('output1/epoch_21_meteor_0.5504_rougel_0.5044.pth', map_location=device))
    model.eval()

    # 加载测试集的真实描述
    with open('data/test_captions.json', 'r') as f:
        test_captions_json = json.load(f)

    # 创建 Tkinter 窗口
    root = tk.Tk()
    root.title("Image Captioning")
    root.geometry("1300x800")  # 增大窗口大小

    # 全局变量
    current_image_path = None
    current_image_label = None
    caption_label = None

    # 显示图片
    def show_image(image_path):
        global current_image_label
        if current_image_label:
            current_image_label.destroy()  # 清除之前的图片

        img = Image.open(image_path)
        img = img.resize((500, 500), Image.Resampling.LANCZOS)  # 增大图片显示尺寸
        img_tk = ImageTk.PhotoImage(img)

        current_image_label = tk.Label(root, image=img_tk)
        current_image_label.image = img_tk  # 保持引用，避免被垃圾回收
        current_image_label.grid(row=0, column=0, padx=20, pady=20)

    # 生成描述
    def generate_caption(image_path):
        global caption_label
        if caption_label:
            caption_label.destroy()  # 清除之前的描述

        # 生成描述
        generated_caption = model.caption(image_path)
        
        # # 获取真实描述
        # image_name = os.path.basename(image_path)
        # true_caption = test_captions_json.get(image_name, "No description available")
        
        # 显示描述
        # caption_text = f"Generated Caption:\n{generated_caption}\n\nTrue Caption:\n{true_caption}"
        caption_text = f"Generated Caption:\n{generated_caption}\n"
        caption_label = tk.Label(
            root,
            text=caption_text,
            wraplength=600,  # 增加描述文本的宽度
            justify="left",
            font=("Arial", 14)  # 设置更大的字体
        )
        caption_label.grid(row=0, column=1, padx=20, pady=20)

    # 上传图片
    def upload_image():
        global current_image_path
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg *.png")]
        )
        if file_path:
            current_image_path = file_path
            show_image(file_path)
            generate_caption(file_path)

    # 创建界面组件
    upload_button = tk.Button(
        root,
        text="Upload Image",
        command=upload_image,
        font=("Arial", 14)  # 设置按钮字体大小
    )
    upload_button.grid(row=1, column=0, columnspan=2, pady=20)

    # 运行 Tkinter 主循环
    root.mainloop()


    # # 加载训练好的模型权重
    # model.load_state_dict(torch.load('output1/epoch_21_meteor_0.5504_rougel_0.5044.pth', map_location=device))
    # model.eval()

    # # 定义结果存储列表
    # results = []

    # # 加载测试集的真实描述
    # with open('data/test_captions.json', 'r') as f:
    #     test_captions_json = json.load(f)

    # # 遍历 images 文件夹中的所有图片
    # images_dir = 'data/images'
    # for image_name in os.listdir(images_dir):
    #     if image_name.endswith('.jpg') or image_name.endswith('.png'):  # 仅处理图片文件
    #         image_path = os.path.join(images_dir, image_name)
            
    #         # 生成描述
    #         generated_caption = model.caption(image_path)
            
    #         # 获取真实描述
    #         true_caption = test_captions_json.get(image_name, "No description available")
            
    #         # 如果真实描述不是 "No description available"，则添加到结果列表中
    #         if true_caption != "No description available":
    #             result = {
    #                 "image": image_name,
    #                 "generated_caption": generated_caption,
    #                 "true_caption": true_caption
    #             }
    #             results.append(result)

    # # 将结果写入 JSON 文件
    # output_file = 'generated_captions_all.json'
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=4)

    # print(f"所有图片的描述已生成并保存到 {output_file}")


else:
    print("Invalid mode. Please choose 1 for training or 2 for generating.")