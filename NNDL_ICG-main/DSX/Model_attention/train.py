import json
import torch
import os
from configuartions import Config
from models import AttentionModel, get_optimizer, PackedCrossEntropyLoss, evaluate_cider
#from model2 import TransformerCaptioningModel ,get_optimizer,PackedCrossEntropyLoss,evaluate_cider
from datasets import create_dataloaders, ImageTextDataset


def main():
    best_test_score = float('-inf')  # 初始化最佳测试得分

    # 加载配置
    config = Config()

    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(config)

    # 加载词汇表文件
    with open('../data/output/vocab.json', 'r') as f:
        vocab = json.load(f)

    # 模型初始化
    model = AttentionModel(
        image_code_dim=config.image_code_dim,
        vocab=vocab,  # 传递词汇表字典
        word_dim=config.word_dim,
        attention_dim=config.attention_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers
    ).to(config.device)

    # 优化器
    optimizer = get_optimizer(model, config)

    # 损失函数
    loss_fn = PackedCrossEntropyLoss().to(config.device)

    # 创建保存权重的文件夹路径
    weights_dir = os.path.join(config.output_folder, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    best_val_score = float('-inf')  # 初始化最佳验证得分

    # 开始训练
    for epoch in range(config.num_epochs):
        model.train()
        for i, (imgs, caps, caplens) in enumerate(train_loader):
            imgs, caps = imgs.to(config.device), caps.to(config.device)
            caplens = caplens.cpu().to(torch.int64)

            optimizer.zero_grad()
            outputs, alphas, _, _, _ = model(imgs, caps, caplens)

            targets = caps[:, 1:]  # 假设targets是captions去除第一个<start>标记后的部分
            print(outputs)
            print(outputs.shape)
            loss = loss_fn(outputs, targets, caplens)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{config.num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        current_test_score = evaluate_cider(test_loader, model, config)
        print(f"Epoch {epoch + 1}: CIDEr-D score = {current_test_score}")

        if current_test_score > best_test_score:
            best_test_score = current_test_score
            best_model_path = os.path.join(weights_dir, f'Attention_model_background_caption_{best_test_score}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path}")

    # 训练完成后的最终评估
    final_test_score = evaluate_cider(test_loader, model, config)
    print(f"Final CIDEr-D score = {final_test_score}")

    # 训练完成后保存模型
    final_model_path = os.path.join(weights_dir, 'AttentionModel.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")


if __name__ == '__main__':
    main()
