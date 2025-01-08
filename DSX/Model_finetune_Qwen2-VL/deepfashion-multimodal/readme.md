- 额外需解压`deepfashion-multimodal.zip`于本文件夹中
- 使用&简单修改`split.py`将images下图片分置为`train_images`和`test_images`两个图片文件夹中，然后把这两个文件夹分放入上一级中的文件夹`test_data`和`train_data`
- 使用&简单修改`transform.py`将两个captions文件修改为符合微调需要的内容结构，然后把这两个文件分放入上一级中的文件夹`test_data`和`train_data`
- 本文件夹结构如下
```bash
deepfashion-multimodal
├── images
├── test_captions.json
├── train_captions.json
├── spilt.py
└── transform.py
```



