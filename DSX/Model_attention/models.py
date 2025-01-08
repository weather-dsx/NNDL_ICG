import torch
import torch.nn as nn
from pycocoevalcap.cider.cider import Cider
import numpy as np
from configuartions import Config
from torchvision.models import resnet101, ResNet101_Weights
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
import json

# 图像编码器模块，包含自注意力机制
class ImageEncoder(nn.Module):
    def __init__(self, finetuned=True, num_heads=8, dropout=0.1):
        super(ImageEncoder, self).__init__()
        # 加载预训练 ResNet-101 模型
        model = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.grid_rep_extractor = nn.Sequential(*(list(model.children())[:-2]))
        # 冻结或解冻特征提取器参数
        for param in self.grid_rep_extractor.parameters():
            param.requires_grad = finetuned
        # 定义自注意力机制
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(model.fc.in_features, num_heads, dropout)

    def forward(self, images):
        # 提取特征图
        features = self.grid_rep_extractor(images)  # 输出形状: (batch_size, num_channels, H, W)
        # 保存原始形状
        orig_shape = features.shape
        # 转换形状为 (sequence_length, batch_size, num_channels)
        features = features.flatten(2).permute(2, 0, 1)
        # 应用自注意力机制
        attention_output, _ = self.attention(features, features, features)
        # 还原形状
        attention_output = attention_output.permute(1, 2, 0).view(orig_shape)
        return attention_output



# 解码器的注意力机制
class AdditiveAttention(nn.Module):
    def  __init__(self, query_dim, key_dim, attn_dim):
        """
        参数：
            query_dim: 查询Q的维度
            key_dim: 键K的维度
            attn_dim: 注意力函数隐藏层表示的维度
        """
        super(AdditiveAttention, self).__init__()
        self.attn_w_1_q = nn.Linear(query_dim, attn_dim)
        self.attn_w_1_k = nn.Linear(key_dim, attn_dim)
        self.attn_w_2 = nn.Linear(attn_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, query, key_value):
        """
        Q K V：Q和K算出相关性得分，作为V的权重，K=V
        参数：
            query: 查询 (batch_size, q_dim)
            key_value: 键和值，(batch_size, n_kv, kv_dim)
        """
        # （2）计算query和key的相关性，实现注意力评分函数
        # -> (batch_size, 1, attn_dim)
        queries = self.attn_w_1_q(query).unsqueeze(1)
        # -> (batch_size, n_kv, attn_dim)
        keys = self.attn_w_1_k(key_value)
        # -> (batch_size, n_kv)
        attn = self.attn_w_2(self.tanh(queries+keys)).squeeze(2)
        # （3）归一化相关性分数
        # -> (batch_size, n_kv)
        attn = self.softmax(attn)
        # （4）计算输出
        # (batch_size x 1 x n_kv)(batch_size x n_kv x kv_dim)
        # -> (batch_size, 1, kv_dim)
        output = torch.bmm(attn.unsqueeze(1), key_value).squeeze(1)
        return output, attn


class AttentionDecoder(nn.Module):
    """
    文本解码器，结合图像特征和文本输入，通过注意力机制生成描述。

    参数:
        image_code_dim: 图像编码的维度。
        vocab_size: 词汇表大小。
        word_dim: 词嵌入的维度。
        attention_dim: 注意力机制的维度。
        hidden_size: GRU隐藏层的大小。
        num_layers: GRU层数。
        dropout: Dropout的概率。
    """
    def __init__(self, image_code_dim, vocab_size, word_dim, attention_dim, hidden_size, num_layers, dropout=0.5):
        super(AttentionDecoder, self).__init__()

        self.embed = nn.Embedding(vocab_size, word_dim)
        self.attention = AdditiveAttention(hidden_size, image_code_dim, attention_dim)
        self.init_state = nn.Linear(image_code_dim, num_layers * hidden_size)
        self.rnn = nn.GRU(word_dim + image_code_dim, hidden_size, num_layers)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """
        初始化网络权重。
        """
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)

    def init_hidden_state(self, image_code, captions, cap_lens):
        """
        根据图像特征和文本长度初始化GRU的隐藏状态。

        参数：
            image_code: 图像特征，形状为 (batch_size, image_code_dim, grid_height, grid_width)。
            captions: 文本输入，形状为 (batch_size, seq_length)。
            cap_lens: 文本长度，形状为 (batch_size,)。
        """
        batch_size, image_code_dim = image_code.size(0), image_code.size(1)
        image_code = image_code.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, image_code_dim)
        sorted_cap_lens, sorted_cap_indices = torch.sort(cap_lens, descending=True)
        captions = captions[sorted_cap_indices]
        image_code = image_code[sorted_cap_indices]
        
        # 初始化GRU隐藏状态
        hidden_state = self.init_state(image_code.mean(dim=1)).view(
            batch_size, self.rnn.num_layers, self.rnn.hidden_size).permute(1, 0, 2)
        
        return image_code, captions, sorted_cap_lens, sorted_cap_indices, hidden_state

    def forward_step(self, image_code, curr_cap_embed, hidden_state):
        """
        执行解码器的前向步骤，生成当前时刻的预测和注意力分数。

        参数：
            image_code: 图像编码，形状为 (batch_size, seq_length, image_code_dim)。
            curr_cap_embed: 当前词的词嵌入，形状为 (batch_size, word_dim)。
            hidden_state: 当前的隐藏状态，形状为 (num_layers, batch_size, hidden_size)。
        """
        # 获取上下文向量和注意力分数
        context, alpha = self.attention(hidden_state[-1], image_code)
        
        # 将上下文向量与当前词的词嵌入拼接并传入GRU
        x = torch.cat((context, curr_cap_embed), dim=-1).unsqueeze(0)
        out, hidden_state = self.rnn(x, hidden_state)
        
        # 通过全连接层生成当前时刻的预测
        preds = self.fc(self.dropout(out.squeeze(0)))
        
        return preds, alpha, hidden_state

    def forward(self, image_code, captions, cap_lens):
        """
        完整的解码过程，处理输入的图像特征和文本描述，生成文本预测。

        参数：
            image_code: 图像编码，形状为 (batch_size, feature_channel, feature_size)。
            captions: 文本描述，形状为 (batch_size, seq_length)。
            cap_lens: 文本长度，形状为 (batch_size,)。
        """
        # 初始化隐藏状态并对文本长度进行排序
        image_code, captions, sorted_cap_lens, sorted_cap_indices, hidden_state = self.init_hidden_state(image_code, captions, cap_lens)
        
        batch_size = image_code.size(0)
        lengths = sorted_cap_lens.cpu().numpy() - 1  # 忽略最后一个词
        max_cap_len = max(cap_lens)
        
        # 初始化预测结果和注意力权重矩阵
        predictions = torch.zeros(batch_size, max_cap_len, self.fc.out_features).to(captions.device)
        alphas = torch.zeros(batch_size, max_cap_len, image_code.shape[1]).to(captions.device)
        
        # 获取词嵌入
        cap_embeds = self.embed(captions)

        for step in range(lengths[0]):
            real_batch_size = np.sum(lengths > step)
            
            # 计算当前时间步的预测和注意力分数
            preds, alpha, hidden_state = self.forward_step(
                image_code[:real_batch_size],
                cap_embeds[:real_batch_size, step, :],
                hidden_state[:, :real_batch_size, :].contiguous())
            
            # 记录结果
            predictions[:real_batch_size, step, :] = preds
            alphas[:real_batch_size, step, :] = alpha
        
        # 填充预测结果，使其与最大文本长度一致
        padded_predictions = torch.zeros(batch_size, max_cap_len, self.fc.out_features).to(predictions.device)
        for i in range(batch_size):
            actual_length = cap_lens[i]
            padded_predictions[i, :actual_length, :] = predictions[i, :actual_length, :]

        return padded_predictions, alphas, captions, lengths, sorted_cap_indices

# AttentionModel 模型
'''
注意：确保 image_code_dim 等参数与 ImageEncoder 的输出匹配

最终 ImageEncoder 的输出形状仍然是 (batch_size, num_channels, height, width)。
这意味着 image_code_dim 应该设置为 num_channels，即 ResNet101 最后一个卷积层的输出通道数。这个值通常为2048，
'''
class AttentionModel(nn.Module):
    def __init__(self, image_code_dim, vocab, word_dim, attention_dim, hidden_size, num_layers):
        super(AttentionModel, self).__init__()
        self.vocab = vocab
        self.encoder = ImageEncoder()
        self.decoder = AttentionDecoder(image_code_dim, len(vocab), word_dim, attention_dim, hidden_size, num_layers)

    def forward(self, images, captions, cap_lens):
        # 打印图像输入形状
        # print("Image input shape:", images.shape)
        image_code = self.encoder(images)
        # 打印编码器输出形状
        # print("Encoder output shape:", image_code.shape)
        output = self.decoder(image_code, captions, cap_lens)
        # 打印解码器输出形状
        # print("Decoder output shape:", output[0].shape)  # Assuming output[0] is the main output
       # print(output[0])
        #print(output[0].shape)
        return output

    def generate_by_beamsearch(self, images, beam_k, max_len):
        vocab_size = len(self.vocab)
        image_codes = self.encoder(images)
        texts = []
        device = images.device
        # 对每个图像样本执行束搜索
        for image_code in image_codes:
            # 将图像表示复制k份
            image_code = image_code.unsqueeze(0).repeat(beam_k, 1, 1, 1)
            # 生成k个候选句子，初始时，仅包含开始符号<start>
            cur_sents = torch.full((beam_k, 1), self.vocab['<start>'], dtype=torch.long).to(device)
            cur_sent_embed = self.decoder.embed(cur_sents)[:, 0, :]
            sent_lens = torch.LongTensor([1] * beam_k).to(device)
            # 获得GRU的初始隐状态
            image_code, cur_sent_embed, _, _, hidden_state = \
                self.decoder.init_hidden_state(image_code, cur_sent_embed, sent_lens)
            # 存储已生成完整的句子（以句子结束符<end>结尾的句子）
            end_sents = []
            # 存储已生成完整的句子的概率
            end_probs = []
            # 存储未完整生成的句子的概率
            probs = torch.zeros(beam_k, 1).to(device)
            k = beam_k
            while True:
                preds, _, hidden_state = self.decoder.forward_step(image_code[:k], cur_sent_embed,
                                                                   hidden_state.contiguous())
                # -> (k, vocab_size)
                preds = nn.functional.log_softmax(preds, dim=1)
                # 对每个候选句子采样概率值最大的前k个单词生成k个新的候选句子，并计算概率
                # -> (k, vocab_size)
                probs = probs.repeat(1, preds.size(1)) + preds
                if cur_sents.size(1) == 1:
                    # 第一步时，所有句子都只包含开始标识符，因此，仅利用其中一个句子计算topk
                    values, indices = probs[0].topk(k, 0, True, True)
                else:
                    # probs: (k, vocab_size) 是二维张量
                    # topk函数直接应用于二维张量会按照指定维度取最大值，这里需要在全局取最大值
                    # 因此，将probs转换为一维张量，再使用topk函数获取最大的k个值
                    values, indices = probs.view(-1).topk(k, 0, True, True)
                # 计算最大的k个值对应的句子索引和词索引
                sent_indices = torch.div(indices, vocab_size, rounding_mode='trunc')
                word_indices = indices % vocab_size
                # 将词拼接在前一轮的句子后，获得此轮的句子
                cur_sents = torch.cat([cur_sents[sent_indices], word_indices.unsqueeze(1)], dim=1)
                # 查找此轮生成句子结束符<end>的句子
                end_indices = [idx for idx, word in enumerate(word_indices) if word == self.vocab['<end>']]
                if len(end_indices) > 0:
                    end_probs.extend(values[end_indices])
                    end_sents.extend(cur_sents[end_indices].tolist())
                    # 如果所有的句子都包含结束符，则停止生成
                    k -= len(end_indices)
                    if k == 0:
                        break
                # 查找还需要继续生成词的句子
                cur_indices = [idx for idx, word in enumerate(word_indices)
                               if word != self.vocab['<end>']]
                if len(cur_indices) > 0:
                    cur_sent_indices = sent_indices[cur_indices]
                    cur_word_indices = word_indices[cur_indices]
                    # 仅保留还需要继续生成的句子、句子概率、隐状态、词嵌入
                    cur_sents = cur_sents[cur_indices]
                    probs = values[cur_indices].view(-1, 1)
                    hidden_state = hidden_state[:, cur_sent_indices, :]
                    cur_sent_embed = self.decoder.embed(
                        cur_word_indices.view(-1, 1))[:, 0, :]
                # 句子太长，停止生成
                if cur_sents.size(1) >= max_len:
                    break
            if len(end_sents) == 0:
                # 如果没有包含结束符的句子，则选取第一个句子作为生成句子
                gen_sent = cur_sents[0].tolist()
            else:
                # 否则选取包含结束符的句子中概率最大的句子
                gen_sent = end_sents[end_probs.index(max(end_probs))]
            texts.append(gen_sent)
        return texts


# 损失函数
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
        # 使用 pack_padded_sequence 来处理变长序列
        # 这里 predictions 和 targets 都需要进行 pack 操作
        # 由于 pack_padded_sequence 需要长度从长到短的序列，这里假设输入已经是这种格式
        packed_predictions = pack_padded_sequence(predictions, lengths, batch_first=True, enforce_sorted=False)[0]
        packed_targets = pack_padded_sequence(targets, lengths, batch_first=True, enforce_sorted=False)[0]

        # 计算损失，忽略填充的部分
        loss = self.loss_fn(packed_predictions, packed_targets)
        return loss


def get_optimizer(model, config):
    """
    获取优化器，为模型的不同部分设置不同的学习速率。
    参数：
        model：训练模型。
        config：包含配置信息的对象，如学习速率等。
    返回：
        配置好地优化器。
    """
    # 为编码器和解码器设置不同的学习速率
    encoder_params = filter(lambda p: p.requires_grad, model.encoder.parameters())
    decoder_params = filter(lambda p: p.requires_grad, model.decoder.parameters())

    # 创建优化器，分别对这两部分参数应用不同的学习速率
    optimizer = optim.Adam([
        {"params": encoder_params, "lr": config.encoder_learning_rate},
        {"params": decoder_params, "lr": config.decoder_learning_rate}
    ])

    return optimizer

# 以下函数是为了展示如何在训练过程中调整学习速率，实际上可能并未使用
def adjust_learning_rate(optimizer, epoch, config):
    """
    调整学习速率，每隔一定轮次减少到原来的十分之一。
    参数：
        optimizer：优化器。
        epoch：当前轮次。
        config：包含配置信息的对象。
    """
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'encoder':
            param_group['lr'] = config.encoder_learning_rate * (0.1 ** (epoch // config.lr_update))
        else:
            param_group['lr'] = config.decoder_learning_rate * (0.1 ** (epoch // config.lr_update))


# CIDEr-D 评估
def filter_useless_words(sent, filterd_words):
    # 去除句子中不参与CIDEr-D计算的符号
    return [w for w in sent if w not in filterd_words]


def evaluate_cider(data_loader, model, config):
    model.eval()
    # 存储候选文本和参考文本
    cands = {}
    refs = {}
    filterd_words = {model.vocab['<start>'], model.vocab['<end>'], model.vocab['<pad>']}
    device = next(model.parameters()).device

    # 加载词汇表并创建反向词汇表
    with open('../data/output/vocab.json', 'r') as f:
        vocab = json.load(f)
    idx_to_word = {idx: word for word, idx in vocab.items()}

    for i, (imgs, caps, caplens) in enumerate(data_loader):
        imgs = imgs.to(device)
        # 通过束搜索生成描述
        preds = model.generate_by_beamsearch(imgs, config.beam_k, config.max_len)
        for j in range(imgs.size(0)):
            img_id = str(i * config.batch_size + j)
            cand_words = [idx_to_word.get(word, '<unk>') for word in preds[j]]
            cand = ' '.join(filter_useless_words(cand_words, filterd_words))
            cands[img_id] = [cand]  # 候选描述
            # 将参考描述（caps[j]）的每个索引转换为单词
            ref_words = [idx_to_word.get(word.item(), '<unk>') for word in caps[j]]
            refs[img_id] = [' '.join(filter_useless_words(ref_words, filterd_words))]  # 参考描述

    # # 在调用 compute_score 之前添加调试信息
    # for key, value in cands.items():
    #     print(f"Key: {key}, Value type: {type(value)}, Value: {value}")
    #     assert isinstance(value, list), f"Value for key {key} is not a list in cands"
    #
    # for key, value in refs.items():
    #     print(f"Key: {key}, Value type: {type(value)}, Value: {value}")
    #     assert isinstance(value, list), f"Value for key {key} is not a list in refs"

    # 计算CIDEr-D得分
    cider_evaluator = Cider()
    score, _ = cider_evaluator.compute_score(refs, cands)
    # score, _ = cider_evaluator.compute_score({'dummy': refs}, {'dummy': cands})

    model.train()
    return score



# encoder = ImageEncoder(Config.embed_size)
# decoder = AttentionDecoder(Config.embed_size, Config.vocab_size, Config.hidden_size, Config.num_layers)
# arctic_model = ARCTIC(encoder, decoder)

# 示例：前馈过程
# images = ...  # 从数据集中获取图像
# captions = ...  # 从数据集中获取对应的文本描述
# 输出 = arctic_model(images, captions)