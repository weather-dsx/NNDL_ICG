# 本仓库内容为22级北邮人工智能专业《神经网络与深度学习》课程设计
## 一、背景介绍
- 采用不同的模型，自动为图片生成流畅关联的自然语言描述，并使用BLEU，METEOR，ROUGE，CIDEr四种评价指标用于模型性能的评判。最后尝试对多模态预训练大模型Qwen2进行微调，比较与自训练模型的差异。
## 二、实验数据
- 原始数据集来源于DeepFashion-MultiModal 数据集中 image 和 textual descriptions 的数据，其中 80% 的数据作为模型的训练集，20% 作为模型的测试集。数据集的 Github Repo 如下：
>  https://github.com/yumingj/DeepFashion-MultiModal
- 我们在此基础上使用老师在群里提供的增量数据：图像集images和切分后的描述文本文件：train_caption.json 和 test_caption.json构成待处理的数据集。
> [谷歌云端硬盘](https://drive.google.com/file/d/1sw-toESmgIZory40qrdeLHzaJ_HpMMtg)
> [阿里云盘(自解压格式压缩文件)](https://www.alipan.com/s/NyZ3XscjepG) (提取码: 40bo)
- 同样按照80%和20%切分train_caption.json文件，划分出训练集train.json与验证集valid.json

## 三、人员分工表

| 部分                  | 探究实现                  | 报告撰写                  | 备注             |
|-----------------------|---------------------------|---------------------------|------------------|
| 1 CNN整体表示+GRU的模型架构 |     张宇昊                  |          张宇昊                 |      无            |
| 2 CNN整体表示+GRU的模型结合GloVe词嵌入 |       董光硕                |            董光硕               |     无             |
| 3 区域表示、自注意+注意力 |           邓圣曦+张宇昊+董光硕           |     邓圣曦                      |        无          |
| 4 区域表示、transformer编码器+transformer解码器 |         邓圣曦              |         邓圣曦                  |         无         |
| 5 大模型微调        |          邓圣曦                   |   张宇昊+邓圣曦              |   无 |
| 6 报告汇总&视频录制        |                         董光硕+张宇昊+邓圣曦              |
