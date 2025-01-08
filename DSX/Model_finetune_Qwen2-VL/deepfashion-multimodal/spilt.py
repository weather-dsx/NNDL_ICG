import json
import shutil
import os

# 读取JSON文件
with open('test_captions.json', 'r', encoding='utf-8') as file:
    train_captions = json.load(file)

# 确保目标文件夹存在
if not os.path.exists('test_images/'):
    os.makedirs('test_images/')

# 遍历JSON数据，复制图片
for image_name, _ in train_captions.items():
    # 提取图片文件名
    base_name = image_name.split('/')[-1]
    source_path = f'images/{base_name}'
    destination_path = f'test_images/{base_name}'

    # 检查源文件是否存在
    if os.path.exists(source_path):
        # 复制文件
        shutil.copy(source_path, destination_path)
        print(f'Copied {base_name} to test_images/')
    else:
        print(f'File {base_name} not found in images/ directory.')

print('All files have been copied.')