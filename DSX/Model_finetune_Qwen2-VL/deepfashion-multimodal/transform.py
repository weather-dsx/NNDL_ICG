import json

# 读取原始的 JSON 文件
with open('test_captions.json', 'r', encoding='utf-8') as file:
    test_captions = json.load(file)

# 准备转换后的数据结构
converted_data = []

for image_name, caption in test_captions.items():
    # 构建新的数据结构，直接使用原始 JSON 中的文件名
    message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"test_data/test_images/{image_name}"  # 直接使用文件名
                    },
                    {"type": "text", "text": "描述一下这个图片"}
                ]
            },
        ]
    converted_data.append(message)

# 将转换后的数据写入新的 JSON 文件
with open('test_data.json', 'w', encoding='utf-8') as file:
    json.dump(converted_data, file, indent=4, ensure_ascii=False)

print("转换完成，已保存到 test_data.json 文件中。")
