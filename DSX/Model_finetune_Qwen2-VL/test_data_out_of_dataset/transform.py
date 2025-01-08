import json
import os

converted_data = []
img_folder = 'img'
# 读取 img 文件夹中的图片文件名
image_names = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]

for image_name in image_names:
    message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"test_data_out_of_dataset/img/{image_name}"  # 直接使用文件名
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
