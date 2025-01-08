import json
import torch
import os
import torch.nn as nn
from configuartions import Config
# from models import AttentionModel, get_optimizer, PackedCrossEntropyLoss, evaluate_cider
from model2_1 import TransformerCaptioningModel, get_optimizer, PackedCrossEntropyLoss, calculate_meteor, calculate_rouge_l
from datasets import create_dataloaders, ImageTextDataset

def check_initial_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} 的初始值分布:")
            print(f"  平均值: {param.data.mean().item():.6f}, 标准差: {param.data.std().item():.6f}")

def save_checkpoint(model, optimizer, epoch_loss, epoch, weights_dir):
    model_path = os.path.join(weights_dir, f'Transformer_model_{epoch+1}.pth')
    optimizer_path = os.path.join(weights_dir, f'optimizer_{epoch+1}.pth')
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    print(f"保存模型和优化器至 {model_path}, {optimizer_path}")

def load_checkpoint(model, optimizer, checkpoint_dir, epoch):
    model_path = os.path.join(checkpoint_dir, f'Transformer_model_{epoch}.pth')
    optimizer_path = os.path.join(checkpoint_dir, f'optimizer_{epoch}.pth')
    
    # 检查是否存在模型和优化器的checkpoint文件
    if os.path.exists(model_path) and os.path.exists(optimizer_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=Config.device))
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=Config.device))
            print(f"成功加载模型和优化器权重: {model_path}, {optimizer_path}")
        except Exception as e:
            print(f"加载失败: {e}")
    else:
        print(f"未找到预训练模型或优化器文件: {model_path}, {optimizer_path}")

def main():
    # 加载配置
    config = Config()

    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(config)

    # 加载词汇表文件
    with open('../data/output/vocab.json', 'r') as f:
        vocab = json.load(f)

    # 模型初始化 - 使用 TransformerCaptioningModel
    model = TransformerCaptioningModel(
        image_code_dim=config.image_code_dim,
        vocab=vocab,
        batch_size=Config.batch_size,
        vocab_size=Config.vocab_size,
        max_len=Config.max_len,
        word_dim=config.word_dim,
        num_heads=config.num_heads,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        device=config.device  # 使用的设备
    )
    model.to(config.device)  # 确保模型在正确的设备上
    # 优化器
    
    #check_initial_weights(model)
    
    optimizer = get_optimizer(model, config)
    loss_fn = PackedCrossEntropyLoss().to(config.device)

    # 创建保存权重的文件夹路径
    weights_dir = os.path.join(config.output_folder, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    best_test_score = float('-inf')
    start_epoch = 0
    # 开始训练
    for epoch in range(start_epoch, config.num_epochs):
        load_checkpoint(model, optimizer, weights_dir, epoch)
        model.train()
        epoch_loss = 0
        for i, (imgs, caps, caplens, _) in enumerate(train_loader):
            imgs, caps = imgs.to(config.device), caps.to(config.device)
            caplens = caplens.to(config.device)  # 确保 caplens 在相同的设备上

            optimizer.zero_grad()

            # 前向传播
            outputs = model(imgs, caps)  # outputs 是由Transformer生成的预测结果
            targets = caps[:, 1:]

            _, predicted_indices = torch.max(outputs, dim=-1)  
            #predictions = torch.softmax(outputs, dim=-1)
            loss = loss_fn(outputs, targets, caplens)  
            
            epoch_loss += loss
            
            loss.backward()

            # 保存更新前的参数
            param_before_update = {name: param.clone().detach() for name, param in model.named_parameters()}
            
            # 更新参数
            optimizer.step()
            '''
            for param in model.parameters():
                if param.grad is not None:
                    print(param.grad.mean().item())

            # 检查参数是否更新
            for name, param in model.named_parameters():
                change = torch.max(torch.abs(param - param_before_update[name])).item()
                if change < 0.00001:
                    print(f"参数 {name} 的变化幅度为 {change:.8f}，可能未更新。")
                else:
                    print(f"参数 {name} 已成功更新，变化幅度为 {change:.6f}。")
            '''
            # 每 10 次打印一次
            if (i + 1) % 100 == 0:
                #print('target',targets[0])
                #print('predicted',predicted_indices[0])
                print(f'Epoch [{epoch + 1}/{config.num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                
                '''
                generated_sentences = model.generate_topk(
                    img = imgs[0].unsqueeze(0),
                    beam_size=5,  # 设置束宽度
                    max_caption_length=config.max_len,
                    temperature = 1.3,
                    repetition_penalty = 1.6
                )
                '''
                generated_sentences = model.generate_gs(
                    img = imgs[0].unsqueeze(0),
                    beam_size=5,  # 设置束宽度
                    max_caption_length=config.max_len,
                    temperature = 1.3,
                    repetition_penalty = 1.6
                )
                '''
                generated_sentences = model.generate_gs_p(
                    img = imgs[0].unsqueeze(0),
                    beam_size=5,  # 设置束宽度
                    max_caption_length=config.max_len,
                    temperature = 1.3,
                    top_p = 0.9,
                    repetition_penalty = 1.6
                )
                '''
                #print(f'Generated Sentences: {generated_sentences}')

                rouge_l_score = calculate_rouge_l(targets[0], generated_sentences,vocab)
                meteor_score = calculate_meteor(targets[0], generated_sentences,vocab)
        
                #print(f"Target Caption: {' '.join(target_words)}")
                #print(f"Generated Caption: {' '.join(generated_sentences)}")
                print(f"ROUGE-L Score: {rouge_l_score:.4f}")
                print(f"METEOR Score: {meteor_score:.4f}")
                print("-" * 50)
        # 保存模型和优化器
        save_checkpoint(model, optimizer, epoch_loss, epoch+1, weights_dir)

    # 训练完成后保存最终模型
    final_model_path = os.path.join(weights_dir, 'TransformerModel.pth')
    torch.save(model.state_dict(), final_model_path)
    torch.save(optimizer.state_dict(), os.path.join(weights_dir, 'final_optimizer.pth'))
    print(f"保存最终模型和优化器至 {final_model_path}")

if __name__ == '__main__':
    main()
