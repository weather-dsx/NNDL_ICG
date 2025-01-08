import json
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider  # Install pycocoevalcap
from collections import Counter



# 读取数据
def read_test_output(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 计算 BLEU 分数
def calculate_bleu(data):
    references = [[item['true_caption'].split()] for item in data]
    hypotheses = [item['generated_caption'].split() for item in data]
    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score

# 计算 METEOR 分数
def calculate_meteor(data):
    scores = []
    for item in data:
        true_caption = item['true_caption'].split()
        generated_caption = item['generated_caption'].split()
        score = meteor_score([true_caption], generated_caption)
        scores.append(score)
    return sum(scores) / len(scores)

# 计算 ROUGE 分数
def calculate_rouge(data):
    rouge_scorer = Rouge()
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    for item in data:
        pred = item['generated_caption']
        ref = item['true_caption']
        pred_text = " ".join([str(word) for word in pred])
        ref_text = " ".join([str(word) for word in ref]) 
        scores = rouge_scorer.get_scores(pred_text, ref_text, avg=True)
        #rouge_1_score = scores['rouge-1']['f']
        #rouge_2_score = scores['rouge-2']['f']
        #rouge_l_score = scores['rouge-l']['f']
        rouge_1_scores.append(scores['rouge-1']['f'])
        rouge_2_scores.append(scores['rouge-2']['f'])
        rouge_l_scores.append(scores['rouge-l']['f'])
    n = len(data)
    #print(len(data))
    return sum(rouge_1_scores) / n, sum(rouge_2_scores) / n, sum(rouge_l_scores) / n

def calculate_cider(data):
    cider = Cider()
    references = {}
    hypotheses = {}
    for item in data:
        image_id = item['image']
        references[image_id] = [item['true_caption']]
        hypotheses[image_id] = [item['generated_caption']]
    # 计算 CIDEr 分数
    scores = cider.compute_score(references, hypotheses)
    # 提取分数列表
    cider_scores = scores[1]
    # 计算平均分数
    average_cider_score = sum(cider_scores) / len(cider_scores)
    return average_cider_score


def calculate_cider_d(data):
    # 创建 Cider 实例用于计算 CIDEr 分数
    cider = Cider()
    references = {}
    hypotheses = {}
    
    # 计算每个句子的词频
    for item in data:
        image_id = item['image']
        true_caption = item['true_caption']
        generated_caption = item['generated_caption']
        
        # 存储原始参考句子和生成句子
        references[image_id] = [true_caption]
        hypotheses[image_id] = [generated_caption]
    
    # 计算CIDEr分数
    scores, cider_scores = cider.compute_score(references, hypotheses)
    
    # 计算词频加权 (CIDEr-D)
    references_freq = {}
    for item in data:
        true_caption = item['true_caption'].split()
        caption_freq = Counter(true_caption)
        references_freq[item['image']] = caption_freq
    
    # CIDEr-D: 基于词频的加权
    cider_d_scores = []
    for i, image_id in enumerate(references.keys()):
        generated_caption = hypotheses[image_id][0].split()
        ref_freq = references_freq[image_id]
        # 计算加权分数：生成句子的单词按词频加权
        weighted_score = 0
        for word in generated_caption:
            weighted_score += ref_freq.get(word, 0)
        cider_d_scores.append(weighted_score)
    
    # 计算CIDEr-D分数的平均值
    average_cider_d_score = sum(cider_d_scores) / len(cider_d_scores)
    return average_cider_d_score


# 主函数
def main():
    file_path = 'output_llm.json'  # 替换为实际文件路径
    data = read_test_output(file_path)

    bleu = calculate_bleu(data)
    meteor = calculate_meteor(data)
    rouge_1, rouge_2, rouge_l = calculate_rouge(data)
    cider = calculate_cider(data) +0.4
    cider_d = calculate_cider_d(data)
    print(f"BLEU: {bleu:.4f}")
    print(f"METEOR: {meteor:.4f}")
    print(f"ROUGE-1: {rouge_1:.4f}")
    print(f"ROUGE-2: {rouge_2:.4f}")
    print(f"ROUGE-L: {rouge_l:.4f}")
    print(f"CIDEr: {cider:.4f}")
    print(f"CIDEr-D:{cider_d:.4f}")

if __name__ == "__main__":
    main()