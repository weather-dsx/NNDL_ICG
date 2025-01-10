import re
from collections import Counter
import torch
import json
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk
nltk.download('wordnet')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 2
with open('data/deepfashion-multimodal/train_captions.json','r') as json_file:  # 获取训练集文件
    traindata = json.load(json_file)
with open('data/deepfashion-multimodal/test_captions.json','r') as json_file:  # 获取测试集文件
    testdata = json.load(json_file)

# 2. 获取所有文本描述
traindescriptions = list(traindata.values())  # 获取训练集数据
testdescriptions = list(testdata.values())  # 获取测试集数据

all_words = []
test_words= []
for description in traindescriptions:
    # 先去除标点符号 然后进行分词
    words = re.findall(r"\b\w+\b|[,.']", description.lower())
    all_words.extend(words)  # 训练集的所有单词

for description in testdescriptions:
    words = re.findall(r"\b\w+\b|[,.']", description.lower())
    test_words.extend(words)  # 测试集的所有单词

# 构建词汇表和映射
word_counts = Counter(all_words)  # 所有单词对应数量的元组
vocab = ['<pad>', '<unk>', '<eos>','<sta>',"'"] + [word for word, _ in word_counts.most_common()]  # 单词表，按频率降序
word_to_idx = {word: idx for idx, word in enumerate(vocab)}  # 构建对应索引

# 3
onehot_encoded_list = []
onehor_encoded_lens = []
test_onehot_encoded_list = []
test_onehor_encoded_lens = []
max_length=0

for description in traindescriptions:
    words = re.findall(r"\b\w+\b|[,.']", description.lower())
    words=['<sta>']+words
    # if len(words)>98:
    #     words = words[0:98]
    words.append('<eos>')

    onehot_encoded = torch.stack([torch.tensor(word_to_idx.get(word, word_to_idx['<unk>'])) for word in words], dim=0)  # 为每一句编码
    onehot_lens = onehot_encoded.size()
    if onehot_encoded.size(0)>max_length:
        max_length=onehot_encoded.size(0)  # 获取最长句子长度
    onehot_encoded_list.append(onehot_encoded)  # 编码列表
    onehor_encoded_lens.append(onehot_lens)  # 长度列表


for description in testdescriptions:
    words = re.findall(r"\b\w+\b|[,.']", description.lower())
    words=['<sta>']+words
    # if len(words)>98:
    #     words = words[0:98]
    words.append('<eos>')

    test_onehot_encoded = torch.stack([torch.tensor(word_to_idx.get(word, word_to_idx['<unk>'])) for word in words], dim=0)  # 为每一句编码
    test_onehot_lens = test_onehot_encoded.size()
    test_onehot_encoded_list.append(test_onehot_encoded)  # 编码列表
    test_onehor_encoded_lens.append(test_onehot_lens)  # 长度列表

# 6. 构建key-one-hot编码的映射字典
key_onehot_dict = dict(zip(traindata.keys(), onehot_encoded_list))  # 图片名与编码构成字典
key_onehot_dict_lens = dict(zip(traindata.keys(), onehor_encoded_lens))  # 图片名与长度构成字典
test_key_dict = dict(zip(testdata.keys(), test_onehot_encoded_list))
test_key_dict_lens = dict(zip(testdata.keys(), test_onehor_encoded_lens))  # 图片名与长度构成字典
# 打印词汇表
# print("词汇表:")
# print(vocab)
# print("key- 字符索引表长度:")
# print(len(key_onehot_dict))  # 10155
#
# # 打印key- one-hot编码 的一个映射
# print("key- 字符索引表的一个映射:")
# print(next(iter(key_onehot_dict.items())))
#
# # 打印
# print("key- 字符索引表size的一个映射:")
# print(next(iter(key_onehot_dict_lens.items())))
#
# print("训练数据中 文本数据最大词数为")
# print(max_length)


def load_glove_embedding(glove_file):
    '''
    加载GloVe词嵌入
    键是词 值是向量
    '''
    embedding_dict = {}
    with open(glove_file, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_dict[word] = vector
    return embedding_dict


def create_embedding_matrix(vocab, embedding_dict, embedding_dim):
    '''
    为给定的词汇表创建一个嵌入矩阵
    对词汇表中的每个词 它检查该词是否在嵌入字典中 如果在 就将矩阵中对应的行设置为该词的嵌入向量
    '''
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for i, word in enumerate(vocab):
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]
        else:
            # 处理特殊标记 可以选择使用GloVe中的对应向量或随机初始化的向量
            if word == '<unk>':
                embedding_matrix[i] = np.random.rand(embedding_dim)
            elif word == '<pad>':
                embedding_matrix[i] = np.zeros(embedding_dim)
            elif word == '<sta>' :
                embedding_matrix[i] = embedding_dict['start']
            elif word == '<eos>' :
                embedding_matrix[i] = embedding_dict['end']
    return embedding_matrix


glove_file = 'glove.6B/glove.6B.300d.txt'
EMBED_DIM = 300
if os.path.exists("glove.6B/glove_vocab_110.pth"):
    print("glove 词典已划分!")
    embedding_matrix_tensor = torch.load("glove.6B/glove_vocab_110.pth")
else:
    print("glove 词典未划分 正在划分...")
    glove_vocab = "glove.6B/glove_vocab_110.pth"
    glove_embeddings = load_glove_embedding(glove_file)
    embedding_matrix = create_embedding_matrix(vocab, glove_embeddings, EMBED_DIM)
    embedding_matrix_tensor = torch.tensor(embedding_matrix)
    torch.save(embedding_matrix_tensor, glove_vocab)
    print("embedding matrix", embedding_matrix)
    print("embedding matrix shape", embedding_matrix.shape)


class CustomDataset(Dataset):
    def __init__(self, key_onehot_dict, key_onehot_dict_lens, max_length, transform=None):
        self.key_onehot_dict = key_onehot_dict
        self.key_onehot_dict_lens = key_onehot_dict_lens
        self.max_length = max_length
        self.keys = list(key_onehot_dict.keys())
        # PyTorch图像预处理流程
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        image = Image.open("data/deepfashion-multimodal/images/"+key).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        onehot_encoded = self.key_onehot_dict[key]
        onehot_len = self.key_onehot_dict_lens[key]

        # 填充到最大长度
        pad_length = self.max_length - onehot_len[0]

        onehot_encoded_max = torch.stack([torch.tensor(word_to_idx['<pad>']) for i in range(pad_length)]) if pad_length > 0 else torch.tensor([])

        caption = torch.cat([onehot_encoded, onehot_encoded_max])
        embeddings = F.embedding(caption, embedding_matrix_tensor)

        return image, caption, onehot_len, embeddings  # 处理为张量的图片、填充为最长长度的用字典翻译过的描述、未填充时的长度、未填充描述对应的词向量


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]  # Remove the last fully connected layer
        self.resnet = nn.Sequential(*modules)
        # self.fc = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)          #  torch.Size([32, 512, 7, 7])
        # features = features.view(features.size(0), -1)
        # features = self.fc(features)

        return features


class DecoderGRU(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, max_length, num_layers=1):
        super(DecoderGRU, self).__init__()
        self.image_code_dim = hidden_size
        # self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size + self.image_code_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.max_length = max_length

        self.flatten = nn.Flatten()
        self.init_state = nn.Linear(512, num_layers * hidden_size)
        self.fc2 = nn.Linear(512 * 49, hidden_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, features, embeddings, length):
        '''
        参数
            features torch.Size([32, 512, 7, 7])
            embeddings torch.Size([32, 120, 300])
            captions torch.Size([32, 120])

        '''

        features, hidden_state = self.init_hidden_state(features)
        # features torch.Size([32, 49, 512])
        # hidden_state torch.Size([1, 32, 256])
        features = self.flatten(features)  # torch.Size([32, 25088])
        features = self.fc2(features)  # torch.Size([32, 512])

        # 拼接图像特征和文本嵌入
        features = features.unsqueeze(1)  # torch.Size([32, 1, 256])
        features = features.expand(-1, MAX_LENGTHS, -1)  # torch.Size([32, 120, 256])

        inputs = torch.cat((embeddings, features), dim=-1)  # torch.Size([32, 120, 556])

        # for i in range(embeddings.shape[1]):        # 120
        for i in range(length):

            output, hidden_state = self.gru(inputs[:, i:i + 1, :], hidden_state)
            # output torch.Size([32, 1, 256])
            # hidden_state torch.Size([1, 32, 256])

            # 使用线性层映射输出维度
            output = self.fc(self.dropout(output))  # torch.Size([32, 1, 110])

            if i > 0:
                predicted = torch.cat((predicted, output), dim=1)  # torch.Size([32, n, 110])
            else:
                predicted = output

        # predicted.shape -> torch.Size([32, 120, 110])

        return predicted

    def init_hidden_state(self, features):
        """
        参数：
            features 图像编码器输出的图像表示
                        (batch_size, image_code_dim, grid_height, grid_width)
        """
        # 将图像网格表示转换为序列表示形式
        batch_size, image_code_dim = features.size(0), features.size(1)  # 32 512

        # -> (batch_size, grid_height, grid_width, image_code_dim)
        features = features.permute(0, 2, 3, 1)
        # -> (batch_size, grid_height * grid_width, image_code_dim)
        features = features.view(batch_size, -1, image_code_dim)

        # features.mean(axis=1).shape  -> torch.Size([32, 512])

        # （2）初始化隐状态
        hidden_state = self.init_state(features.mean(axis=1))  # hidden_state size torch.Size([32, 512])

        hidden_state = hidden_state.view(
            batch_size,
            self.gru.num_layers,
            self.gru.hidden_size).permute(1, 0, 2)  # hidden_state size torch.Size([1, 32, 512])

        return features, hidden_state

    def generate_(self, features, captions):
        '''
        测试
            embeddings torch.Size([300])
            features torch.Size([1, 512, 7, 7])
            captions torch.Size([1]) 一般是开始符
        '''
        embeddings = F.embedding(captions, embedding_matrix_tensor.to(device))
        embeddings = embeddings.unsqueeze(0).unsqueeze(0)

        features, hidden_state = self.init_hidden_state(features)
        # features torch.Size([1, 49, 512])
        # hidden_state torch.Size([1, 1, 512])
        features = self.flatten(features)  # torch.Size([1, 25088])
        features = self.fc2(features)  # torch.Size([1, 512])

        # 拼接图像特征和文本嵌入
        features = features.unsqueeze(1)  # torch.Size([1, 1, 512])

        captions = captions.unsqueeze(0)  # tensor([3])

        for _ in range(MAX_LENGTHS):
            # print(f'epoch\t{_+1}')
            inputs = torch.cat((embeddings, features), dim=-1)  # torch.Size([1, 1, 812])

            output, hidden_state = self.gru(inputs.float(), hidden_state.float())
            # hidden_state torch.Size([1, 1, 512])
            # output torch.Size([1, 1, 512])

            output = self.fc(output)
            # torch.Size([1, 1, 110])

            _, predicted = output[0][0].max(0)
            predicted = predicted.unsqueeze(0)  # torch.Size([1])

            captions = torch.cat((captions, predicted), dim=0)
            # Append the predicted word to the captions

            embeddings = F.embedding(predicted[0], embedding_matrix_tensor.to(device))
            embeddings = embeddings.unsqueeze(0).unsqueeze(0)

            # Stop if the end token is generated
            if predicted.item() == vocab.index('<eos>'):
                break

        return captions

    def generate_by_beamsearch(self, features, captions, beam_k=3):
        '''
        束搜索

        在每个时间步
        我们首先清空all_candidates列表 然后遍历candidates中的每个序列
        对每个序列尝试添加一个新的词 生成新的序列 并将新的序列添加到all_candidates中
        然后 我们会对all_candidates中的所有序列进行评分 并选择得分最高的beam_k个序列
        将这些序列赋值给candidates 作为下一个时间步的候选序列
        这个过程会一直持续到生成的序列达到预定的长度 或者所有的候选序列都已经结束
        '''
        embeddings = F.embedding(captions, embedding_matrix_tensor.to(device))
        embeddings = embeddings.unsqueeze(0).unsqueeze(0)

        features, hidden_state = self.init_hidden_state(features)
        # features torch.Size([1, 49, 512])
        # hidden_state torch.Size([1, 1, 512])
        features = self.flatten(features)  # torch.Size([1, 25088])
        features = self.fc2(features)  # torch.Size([1, 512])

        # 拼接图像特征和文本嵌入
        features = features.unsqueeze(1)  # torch.Size([1, 1, 512])

        # Initialize the list of candidate sequences
        candidates = [(captions.unsqueeze(0), 0, hidden_state)]

        for _ in range(MAX_LENGTHS):
            # Store all the candidates for the next step
            all_candidates = []

            for i in range(len(candidates)):

                seq, score, hidden_state = candidates[i]

                if seq[-1] == vocab.index('<eos>') or seq[-1] == vocab.index('<pad>'):
                    all_candidates.append((seq, score, hidden_state))
                    continue
                else:
                    embeddings = F.embedding(seq[-1], embedding_matrix_tensor.to(device))
                    embeddings = embeddings.unsqueeze(0).unsqueeze(0)

                    inputs = torch.cat((embeddings, features), dim=-1)  # torch.Size([1, 1, 812])

                    output, hidden_state_new = self.gru(inputs.float(), hidden_state.float())
                    # hidden_state torch.Size([1, 1, 512])
                    # output torch.Size([1, 1, 512])

                    output = self.fc(output)
                    # torch.Size([1, 1, 110])

                    output = nn.functional.softmax(output, dim=2)  # torch.Size([1, 1, 110])

                    # Get the top k predictions
                    topk_probs, topk_ids = torch.topk(output[0, 0], beam_k)
                    # torch.Size([5]), torch.Size([5])

                    for k in range(beam_k):
                        wordID = topk_ids[k]
                        prob = topk_probs[k]

                        # Add the new word to the sequence and update the score
                        seq_new = seq.clone().detach()
                        seq_new = torch.cat([seq_new, wordID.unsqueeze(0)])
                        score_new = score - torch.log(prob)

                        all_candidates.append((seq_new, score_new, hidden_state_new))

            # Sort all candidates by score
            ordered = sorted(all_candidates, key=lambda tup: tup[1])

            # Select the top k candidates
            candidates = ordered[:beam_k]

        # Return the sequence with the highest score
        return candidates[0][0]


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,max_length):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderGRU(embed_size, hidden_size, vocab_size,max_length=max_length)
    def forward(self, images, embeddings, length):
        features = self.encoder(images)
        outputs = self.decoder(features, embeddings, length)
        return outputs


transform2ResNet18 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
MAX_LENGTHS = 120
embed_size = 300
hidden_size = 256
vocab_size = len(vocab)
BATCH_SIZE = 20
EPOCHS = 20

model = ImageCaptioningModel(embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size,max_length=MAX_LENGTHS).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# 创建数据集实例
custom_dataset = CustomDataset(key_onehot_dict, key_onehot_dict_lens, max_length=MAX_LENGTHS, transform=transform2ResNet18)
test_dataset = CustomDataset(test_key_dict, test_key_dict_lens, max_length=MAX_LENGTHS, transform=transform2ResNet18)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
# 创建 DataLoader 实例
data_loader = DataLoader(dataset=custom_dataset, batch_size=BATCH_SIZE, shuffle=True)


def evaluate(image_path):
    model.eval()
    ima = Image.open("data/deepfashion-multimodal/images/" + image_path)
    image = ima.convert("RGB")
    image = transform2ResNet18(image).unsqueeze(0).to(
        device)

    with torch.no_grad():
        features = model.encoder(image)

    start_token = vocab.index('<sta>')
    captions = torch.tensor([start_token], dtype=torch.long).unsqueeze(0).to(
        device)
    captions = captions[0]

    with torch.no_grad():
        captions = model.decoder.generate_(features, captions[-1])

    captions = captions.cpu().numpy()

    caption = ' '.join([vocab[i] for i in captions[1:]])
    print(caption)
    plt.imshow(ima)
    plt.show()

5.
def compute_rouge_l_score(pred, reference):
    rouge_scorer = Rouge()
    scores = rouge_scorer.get_scores(pred[1:-1], reference, avg=True)
    rouge_l_score = scores['rouge-l']['f']
    return rouge_l_score


def compute_meteor_score(pred, reference):
    reference = [reference.split()]
    meteor_score_value = meteor_score(reference, pred.split()[1:-1])

    return meteor_score_value


if __name__ == '__main__':
    max1 = 0
    max2 = 0
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        # 存储所有预测和参考值以计算评估指标
        all_predictions = []
        all_references = []

        for batch in tqdm(data_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}'):
            images = batch[0]
            captions = batch[1]
            lengths = batch[2]
            length = torch.max(lengths[0])
            embeddings = batch[3]

            images = images.to(device)
            captions = captions.long().to(device)
            embeddings = embeddings.to(device).long()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, embeddings, length).type(torch.FloatTensor)
            outputs = outputs.permute(0, 2, 1).to(device)

            # Calculate loss (ignoring padding)
            loss = criterion(outputs[:, :, :-1], captions[:, 1:length])

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        # 禁用梯度计算
        with torch.no_grad():
            for batch in test_loader:
                images = batch[0]
                captions = batch[1]
                lengths = batch[2]
                length = torch.max(lengths[0])
                embeddings = batch[3]

                images = images.to(device)
                captions = captions.long().to(device)
                embeddings = embeddings.to(device).long()

                outputs = model(images, embeddings, length).type(torch.FloatTensor)
                outputs = outputs.permute(0, 2, 1).to(device)
                predictions = outputs.argmax(dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_references.extend(captions.cpu().numpy())

        average_loss = total_loss / len(data_loader)
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {average_loss:.4f}')
        losses.append(average_loss)

        # 计算评价指标
        rouge_l_scores = []
        meteor_scores = []

        for pred, ref in zip(all_predictions, all_references):
            pred_text = " ".join([str(word) for word in pred])
            ref_text = " ".join([str(word) for word in ref])

            rouge_l = compute_rouge_l_score(pred_text, ref_text)
            meteor = compute_meteor_score(pred_text, ref_text)

            rouge_l_scores.append(rouge_l)
            meteor_scores.append(meteor)

        avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
        avg_meteor = sum(meteor_scores) / len(meteor_scores)
        print(f'Epoch [{epoch + 1}/{EPOCHS}], ROUGE-L: {avg_rouge_l:.4f}, METEOR: {avg_meteor:.4f}')
        if max1 < avg_rouge_l or max2 < avg_meteor:
            max1 = max(avg_rouge_l,max1)
            max2 = max(avg_rouge_l,max2)
            torch.save(model.state_dict(), f'epoch{epoch+1}-ROUGE-L{avg_rouge_l:.4f}-METEOR{avg_meteor:.4f}.pth')

        image_path = random.choice(list(traindata.keys()))
        evaluate(image_path)
        model.train()
