- 解压`deepfashion-multimodal.zip`把images文件夹替换到data下面来。
文件夹结构如下
```
data
├── images
├── train_captions.json
├── test_captions.json
├── train.json
└── valid.json
```

- CNN-GRU 模型文件夹结构
```
Model
├── data
├── output1                        
├── cnngru.py
├── dataset.py
└── makedata.py
```
dataset和makedata为数据处理，cnngru为主要实现，output1为保存预训练模型
具体介绍参考ipynb文档