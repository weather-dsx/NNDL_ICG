import torch
import math
import json
import logging
import torch.optim as optim
from pycocoevalcap.cider.cider import Cider
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
from torch.nn import Transformer
from configuartions import Config
import torch.nn.functional as F
import logging
import nltk
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

#nltk.download('wordnet')
# 配置日志记录
logging.basicConfig(
    filename="generation_log.txt",  # 日志文件名
    level=logging.DEBUG,  # 日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
    filemode="w"  # 写模式，覆盖旧日志
)


def sinusoidal_position_encoding(max_len, d_model):
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # Add batch dimension

def has_duplicate_words_in_seq(seq):
    """
    检查序列中是否存在重复单词（针对当前句子内）
    """
    word_list = [word.item() for word in seq[0]]  # 将序列中的词转换为普通列表方便检查
    seen = set()
    for word in word_list:
        if word in seen:
            return True
        seen.add(word)
    return False
    
class ImageEncoder(nn.Module):
    def __init__(self, finetuned=True, device='cuda'):
        super(ImageEncoder, self).__init__()
        model = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.grid_rep_extractor = nn.Sequential(*(list(model.children())[:-2]))
        for param in self.grid_rep_extractor.parameters():
            param.requires_grad = finetuned
        
        # 将模型移动到指定设备
        self.device = device
        self.to(self.device)

    def forward(self, images):
        images = images.to(self.device)  # 确保输入数据也在正确的设备上
        features = self.grid_rep_extractor(images)
        return features  # 输出网格表示(维度为[4,512,8,8]的结构，512为通道数&特征维度)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, max_len):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_len = max_len

        # 定义 Transformer 层
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=0.1, activation='relu')
            for _ in range(num_layers)
        ])
        
        # 生成固定位置编码矩阵
        self.register_buffer('positional_encoding', sinusoidal_position_encoding(max_len, d_model)) 

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        positional_encoding = self.positional_encoding[:, :seq_len, :]  # (1, seq_len, d_model)
        x = x + positional_encoding
        for layer in self.encoder_layers:
            x = layer(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, max_length, vocab, vocab_size, word_dim, num_channels=2048, num_heads=8, hidden_size=512, num_layers=6, dropout=0.1, device='cuda'):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_dim, device=device).to(device) 
        self.transformer_decoder = Transformer(
            d_model=num_channels, 
            nhead=num_heads, 
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        ).to(device)
        self.fc_out = nn.Linear(word_dim, vocab_size).to(device)
        self.dropout = nn.Dropout(p=dropout).to(device)
        self.device = device
        self.to(self.device)

    def forward(self, tgt, memory, cap_lens):
        tgt = tgt.to(self.device)
        tgt = self.embedding(tgt)  # tgt变形为(batchsize, 描述文本长度, 词嵌入维度) 
        #tgt = self.dropout(self.embedding(tgt))

        output = self.transformer_decoder(tgt, memory * Config.scale_factor)  # 解码结果output
        output = self.fc_out(self.dropout(output))  # 形状为(batchsize, 描述文本长度, 词汇表大小) 
        max_cap_len = max(cap_lens)
        batch_size = len(cap_lens)
        padded_output = torch.zeros(batch_size, max_cap_len, self.fc_out.out_features).to(self.device)
        for i in range(batch_size):
            actual_length = cap_lens[i]
            padded_output[i, :actual_length, :] = output[i, :actual_length, :]
        return padded_output

class TransformerCaptioningModel(nn.Module):
    def __init__(self, image_code_dim, vocab, batch_size, vocab_size, max_len, word_dim, num_heads=8, hidden_size=512, num_layers=6, dropout=0.1, device='cuda'):
        super(TransformerCaptioningModel, self).__init__()
        self.device = device
        self.vocab = vocab
        self.encoder = ImageEncoder(finetuned=True, device=self.device)
        self.transformer_encoder = TransformerEncoder(d_model=image_code_dim, max_len=max_len, num_heads=num_heads, num_layers=num_layers)
        self.decoder = TransformerDecoder(num_channels=image_code_dim, max_length=max_len, vocab=vocab, vocab_size=vocab_size, word_dim=word_dim, num_heads=num_heads, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, device=device)

    def forward(self, imgs, caps=None):
        imgs = imgs.to(self.device)
        if caps is not None:
            caps = caps.to(self.device)  
    
        image_code = self.encoder(imgs)  
        image_code = image_code.flatten(2).permute(0, 2, 1)  
        memory = self.transformer_encoder(image_code)  
        
        cap_lens = torch.sum(caps != 0, dim=-1) 
        output = self.decoder(caps, memory, cap_lens)
        
        return output
    def generate_gs(self, img, beam_size=3, max_caption_length=20, temperature=1.0, repetition_penalty=1.2):
        """
        使用贪心搜索生成图像字幕，并通过重复检测抑制生成重复词。

        :param img: 输入图像张量，形状为 (1, C, H, W)，需要已经在正确的设备上
        :param beam_size: 束宽，即每一步保留的候选词数量（贪心搜索时，通常是1）
        :param max_caption_length: 生成的字幕最大长度
        :param temperature: 控制生成多样性的温度参数
        :param repetition_penalty: 防止生成重复词的惩罚系数（保留参数）
        :return: 生成的字幕文本（单词列表形式）
        """
        # 将词表索引转换为单词
        inverted_vocab = {value: key for key, value in self.vocab.items()}
        #print(self.vocab)
        with torch.no_grad():
            # 编码图像特征
            image_code = self.encoder(img).flatten(2).permute(0, 2, 1)
            memory = self.transformer_encoder(image_code)
    
            # 初始化贪心搜索的开始 token
            start_token = torch.tensor([self.vocab["<start>"]], device=self.device).unsqueeze(0)
            end_token = self.vocab["<end>"]
    
            # 初始化输入序列
            seq = start_token
            generated_caption = []
    
            for _ in range(max_caption_length):
                # 将当前序列传递到解码器
                output = self.decoder(seq, memory, [seq.size(1)])
                logits = output.squeeze(0)[-1]  # 获取最后一步预测的词分布
                adjusted_logits = logits / temperature
                
                # 防止重复生成词，将已经生成的词的概率设置为 -inf
                for token in seq.squeeze(0):
                    adjusted_logits[token] = float('-inf')
                
                probs = torch.softmax(adjusted_logits, dim=0)
    
                # 获取概率最大的词（贪心搜索）
                next_token = torch.argmax(probs).view(1, -1)  # (1, 1)
    
                # 拼接新词到序列
                seq = torch.cat([seq, next_token], dim=1)
    
                # 记录生成的词
                next_word = inverted_vocab.get(next_token.item(), "<un>")
                generated_caption.append(next_word)
    
                # 如果生成的词是结束符，提前结束生成
                if next_token.item() == end_token:
                    break
    
        return generated_caption



    def generate_gs_p(self, img, beam_size=3, max_caption_length=20, temperature=1.0, top_p=0.9, repetition_penalty=1.2):
        """
        使用 Top-p 采样生成图像字幕，并通过 n-gram 重复检测抑制重复生成。

        :param img: 输入图像张量，形状为 (1, C, H, W)，需要已经在正确的设备上
        :param beam_size: 束宽（保留参数，但 Top-p 采样不使用）
        :param max_caption_length: 生成的字幕最大长度
        :param temperature: 控制生成多样性的温度参数
        :param top_p: 控制采样范围的累计概率阈值
        :param repetition_penalty: 保留参数，不再使用
        :return: 生成的字幕文本（单词列表形式）
        """
        # 将词表索引转换为单词
        inverted_vocab = {value: key for key, value in self.vocab.items()}
        
        def mask_repeated_ngrams(seq, logits, n=2):
            """
            根据 n-gram 重复检测对 logits 进行掩码处理，防止重复生成。
            :param seq: 当前生成序列的词索引张量
            :param logits: 当前 logits
            :param n: n-gram 的长度
            """
            if seq.size(1) < n:
                return logits
            ngrams = [tuple(seq[0, i:i+n].tolist()) for i in range(seq.size(1) - n + 1)]
            repeated_ngrams = set(ngram for ngram in ngrams if ngrams.count(ngram) > 1)
            if repeated_ngrams:
                for ngram in repeated_ngrams:
                    # 获取重复 n-gram 的下一个可能词，并掩码
                    next_word = ngram[-1]
                    logits[next_word] = float('-inf')
            return logits
        
        with torch.no_grad():
            # 编码图像特征
            image_code = self.encoder(img).flatten(2).permute(0, 2, 1)
            memory = self.transformer_encoder(image_code)
    
            # 初始化贪心搜索的开始 token
            start_token = torch.tensor([self.vocab["<start>"]], device=self.device).unsqueeze(0)
            end_token = self.vocab["<end>"]
    
            # 初始化输入序列
            seq = start_token
            generated_caption = []
    
            for _ in range(max_caption_length):
                # 将当前序列传递到解码器
                output = self.decoder(seq, memory, [seq.size(1)])
                logits = output.squeeze(0)[-1]  # 获取最后一步预测的词分布
                adjusted_logits = logits / temperature
                
                # 防止 n-gram 重复生成
                adjusted_logits = mask_repeated_ngrams(seq, adjusted_logits, n=5)
                probs = torch.softmax(adjusted_logits, dim=0)

                # 按概率排序
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)

                # 保留累计概率小于 top_p 的词
                filtered_indices = cumulative_probs <= top_p
                filtered_probs = sorted_probs * filtered_indices.float()
                filtered_probs /= filtered_probs.sum()  # 归一化概率

                # 从过滤后的概率分布中采样
                next_token = torch.multinomial(filtered_probs, num_samples=1)
                next_token = sorted_indices[next_token]  # 映射回原始词表索引
    
                # 拼接新词到序列
                seq = torch.cat([seq, next_token.unsqueeze(0)], dim=1)
                #print('seq',seq)
                # 记录生成的词
                next_word = inverted_vocab.get(next_token.item(), "<un>")
                generated_caption.append(next_word)
    
                # 如果生成的词是结束符，提前结束生成
                if next_token.item() == end_token:
                    break
    
        return generated_caption


    def generate_topk(self, img, beam_size=3, max_caption_length=20, temperature=1.0, repetition_penalty=1.2):
        """
        使用束搜索生成图像字幕。
        
        :param img: 输入图像张量，形状为 (1, C, H, W)，需要已经在正确的设备上
        :param beam_size: 束宽，即每一步保留的候选词数量
        :param max_caption_length: 生成的字幕最大长度
        :param temperature: 控制生成多样性的温度参数
        :param repetition_penalty: 重复惩罚参数
        :return: 生成的字幕文本（单词列表形式）
        """
        # 将词表索引转换为单词
        inverted_vocab = {value: key for key, value in self.vocab.items()}
        
        with torch.no_grad():
            # 编码图像特征
            image_code = self.encoder(img).flatten(2).permute(0, 2, 1)
            memory = self.transformer_encoder(image_code)
    
            # 初始化束搜索变量
            start_token = torch.tensor([self.vocab["<start>"]], device=self.device).unsqueeze(0)
            end_token = self.vocab["<end>"]
            beam_seq = [[start_token, 0.0, set()]]  # 每个元素是 (序列, 分数, 已出现的单词集合)
            completed_seqs = []
            for _ in range(max_caption_length):
                all_candidates = []
                # 对当前束中的每个序列生成下一步的候选
                for seq, score, seen_tokens in beam_seq:
                    tgt = seq  # 当前序列 (1, seq_len)
                    output = self.decoder(tgt, memory, [seq.size(1)])  # 获取解码器输出
                    logits = output.squeeze(0)[-1]  # 取最后一步预测的词分布 (vocab_size,)
                    adjusted_logits = logits / temperature
                    probs = torch.softmax(adjusted_logits, dim=0)
                    # 获取 top-k 候选词及其概率
                    top_probs, top_indices = probs.topk(k=beam_size, dim=0)
                    for i in range(beam_size):
                        token = top_indices[i].item()
                        new_seen_tokens = seen_tokens.copy()  # 复制已出现的单词集合
                        new_seen_tokens.add(token)  # 添加当前候选词到集合
                        # 如果候选词重复，则减少得分
                        if token in seen_tokens:
                            score -= 1  # 如果重复，则得分减1
                        # 防止重复生成词
                        if token in seq.squeeze(0):  # 如果候选词重复
                            top_probs[i] /= repetition_penalty
                        
                        next_token = top_indices[i].view(1, -1)  # (1, 1)
                        new_seq = torch.cat([seq, next_token], dim=1)  # 拼接新词
                        new_score = score + top_probs[i].item()  # 累加概率得分
                        all_candidates.append([new_seq, new_score, new_seen_tokens])  # 也传递新的已见单词集合
    
                # 根据分数对候选进行排序
                all_candidates = sorted(all_candidates, key=lambda x: -x[1])
                beam_seq = []
    
                # 保留束宽数量的最佳候选
                for cand_seq, cand_score, _ in all_candidates[:beam_size]:
                    if cand_seq[0, -1].item() == end_token:  # 若以结束符号结尾，加入完成序列
                        completed_seqs.append([cand_seq, cand_score])
                    else:
                        beam_seq.append([cand_seq, cand_score, _])  # 传递已见单词集合
    
                if len(beam_seq) == 0:  # 若束为空，提前结束
                    break
    
            # 若无完整序列，以当前束中分数最高的序列为最终结果
            if len(completed_seqs) == 0:
                best_seq, _ = max(beam_seq, key=lambda x: x[1])
            else:
                best_seq, _ = max(completed_seqs, key=lambda x: x[1])
    
            # 解码序列为文本，去除起始符号和结束符号
            caption = [
                inverted_vocab.get(token.item(), "<un>")
                for token in best_seq[0, 1:] if token.item() != end_token
            ]
            
            return caption
    
class PackedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(PackedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, lengths):
        """
        计算交叉熵损失，排除填充的部分。
        参数：
            predictions：模型的预测结果，形状为 (batch_size, max_length, vocab_size)。
            targets：实际的文本描述，形状为 (batch_size, max_length)。
            lengths：每个描述的实际长度。
        """
        packed_predictions = pack_padded_sequence(predictions, lengths.cpu(), batch_first=True, enforce_sorted=False)[0]
        packed_targets = pack_padded_sequence(targets, lengths.cpu(), batch_first=True, enforce_sorted=False)[0]

        loss = self.loss_fn(packed_predictions, packed_targets)
        return loss

def get_optimizer(model, config):
    """
    获取优化器，设置不同部分的学习率，并确保模型参数在正确的设备上。

    参数：
        model：训练模型。
        config：包含配置信息的对象，如学习速率等。
    
    返回：
        配置好的优化器。
    """
    encoder_params = filter(lambda p: p.requires_grad, model.encoder.parameters())
    transformer_encoder_params = filter(lambda p: p.requires_grad, model.transformer_encoder.parameters())
    decoder_params = filter(lambda p: p.requires_grad, model.decoder.parameters())

    # 使用SGD优化器
    optimizer = optim.SGD([
        {"params": encoder_params, "lr": config.encoder_learning_rate, "momentum": 0.9},  # 设置动量
        {"params": transformer_encoder_params, "lr": config.encoder_learning_rate, "momentum": 0.9},
        {"params": decoder_params, "lr": config.decoder_learning_rate, "momentum": 0.9}
    ], weight_decay=config.weight_decay)  # 添加权重衰减以防止过拟合

    return optimizer


def filter_useless_words(sent, filterd_words):
    return [w for w in sent if w not in filterd_words]

def calculate_rouge_l(target, generated, vocab):
    """
    计算ROUGE-L分数

    :param target: 目标字幕，数字列表形式
    :param generated: 生成的字幕，数字列表形式
    :param vocab: 词汇表，用于将数字索引转换为单词
    :return: ROUGE-L分数
    """
    #print(vocab)    
    target = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
    generated = generated.cpu().numpy() if isinstance(generated, torch.Tensor) else generated
    
    inverted_vocab = {value: key for key, value in vocab.items()}
    # 生成目标字幕的单词列表，参考生成字幕的方式，去除<pad> token
    target_words = [
        inverted_vocab.get(token, "<un>")  # 获取目标token的对应单词
        for token in target if token != vocab["<pad>"]  # 忽略<pad> token
    ]
    #print(target_words)
    generated_words =  generated 
    #print(generated_words)
    # 使用ROUGE库计算ROUGE-L分数
    rouge = Rouge()
    scores = rouge.get_scores(' '.join(generated_words), ' '.join(target_words))
    
    # 提取ROUGE-L分数
    rouge_l_score = scores[0]['rouge-l']['f']
    return rouge_l_score

def calculate_meteor(target, generated, vocab):
    """
    计算METEOR分数

    :param target: 目标字幕，数字列表形式
    :param generated: 生成的字幕，数字列表形式
    :param vocab: 词汇表，用于将数字索引转换为单词
    :return: METEOR分数
    """
    #print(vocab)
    target = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
    generated = generated.cpu().numpy() if isinstance(generated, torch.Tensor) else generated
    
    # 将数字列表转换为单词列表
    inverted_vocab = {value: key for key, value in vocab.items()}
    target_words = [
        inverted_vocab.get(token, "<un>")  # 获取目标token的对应单词
        for token in target if token != vocab["<pad>"]  # 忽略<pad> token
    ]
    print([target_words])    
    print(generated)
    generated_words = generated
    #print(generated_words)
    # 计算METEOR分数
    score = meteor_score([target_words], generated_words)
    return score

