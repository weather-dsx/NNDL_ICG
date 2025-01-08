import json
import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from util.vision_util import process_vision_info
import torch

# Define the model directory and load the model and processor
model_dir = "train_output/20241224152702/"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_dir, padding_side="left")

# Load test data from test_data.json
with open("test_data/test_data.json", "r") as f:
    test_data = json.load(f)

# Load true captions from test_captions.json
with open("test_data/test_captions.json", "r") as f:
    true_captions = json.load(f)

# Save results to test_output.json
output_data = []
for messages in test_data:
    # Extract image name from messages
    content = messages[0]["content"]  # Access the first dictionary in the list
    image_path = next(item["image"] for item in content if item["type"] == "image")
    image_name = os.path.basename(image_path)  # Extract image file name

    true_caption = true_captions.get(image_name, "")  # Get the true caption for the image

    # Process the input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([messages])  # Single message processing
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=64)
    full_caption = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Extract only the assistant's response
    generated_caption = full_caption.split("assistant\n")[-1].strip()
    print(generated_caption)
    # Store results
    result = {
        "image": image_name,
        "generated_caption": generated_caption,
        "true_caption": true_caption,
    }
    output_data.append(result)
    print(result)

# Save to output file
with open("test_data/test_output.json", "w") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("Output saved to test_output.json")
