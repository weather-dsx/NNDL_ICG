from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from util.vision_util import process_vision_info
from pprint import pprint
import json
import os

model_dir = "./Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, 
    torch_dtype="auto", 
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_dir, padding_side="left")
with open("test_data_out_of_dataset/test_data.json", "r") as f:
    test_data = json.load(f)

output_data = []
for messages in test_data:
    content = messages[0]["content"]  
    image_path = next(item["image"] for item in content if item["type"] == "image")
    image_name = os.path.basename(image_path)  

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([messages]) 
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=64)
    full_caption = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    generated_caption = full_caption.split("assistant\n")[-1].strip()
    print(generated_caption)
    result = {
        "image": image_name,
        "generated_caption": generated_caption,
    }
    output_data.append(result)
    print(result)

with open("test_data_out_of_dataset/test_output_o.json", "w") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("Output saved to test_output_o.json")
