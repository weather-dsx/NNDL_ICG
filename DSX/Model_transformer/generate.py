import json
import torch
import os
from configuartions import Config
from model2_1 import TransformerCaptioningModel
from datasets import create_dataloaders

def load_model(model, checkpoint_dir, epoch):
    model_path = os.path.join(checkpoint_dir, f'TransformerModel.pth')
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=Config.device))
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print(f"Model file not found: {model_path}")
def generate_captions(test_loader, model, vocab, config):
    inverted_vocab = {value: key for key, value in model.vocab.items()}  
    
    results = []  
    for imgs, caps, caplens, img_ids in test_loader:
        imgs = imgs.to(config.device)
        caps = caps.to(config.device)
        #print(img_ids)
        #print(imgs.size(0))       
        # Generate caption for each image in the batch
        for i in range(imgs.size(0)):
            image = imgs[i].unsqueeze(0)  # Single image

            generated_sentences = model.generate_gs(
                img=image,
                beam_size=5,  # Set beam width
                max_caption_length=config.max_len,
                temperature=1.3,
                repetition_penalty=1.6
            )
    
            true_caption_indices = caps[i].tolist()
            true_caption_text = " ".join(
                [inverted_vocab.get(idx, "<un>") for idx in true_caption_indices ]
            )
            new_result = {
                "image": img_ids[i],  # Placeholder for the image filename
                "generated_caption": " ".join(generated_sentences),
                "true_caption": true_caption_text
            }
            print(new_result)
            results.append(new_result)
    return results

def main():
    # Load configuration
    config = Config()

    # Create data loaders
    _, test_loader = create_dataloaders(config)
    
    # 加载词汇表文件
    with open('../data/output/vocab.json', 'r') as f:
        vocab = json.load(f)
        
    model = TransformerCaptioningModel(
        image_code_dim=config.image_code_dim,
        vocab=vocab,
        batch_size=config.batch_size,
        vocab_size=config.vocab_size,
        max_len=config.max_len,
        word_dim=config.word_dim,
        num_heads=config.num_heads,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        device=config.device
    )
    model.to(config.device)

    checkpoint_dir = os.path.join(config.output_folder, 'weights')
    epoch_to_load = config.num_epochs  # Replace with specific epoch if needed
    load_model(model, checkpoint_dir, epoch_to_load)

    results = generate_captions(test_loader, model, vocab, config)

    output_path = os.path.join(config.output_folder, 'generated_captions.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Generated captions saved to {output_path}")

if __name__ == '__main__':
    main()
