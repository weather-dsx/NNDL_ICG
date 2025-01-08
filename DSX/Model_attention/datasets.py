import os
import json
from collections import Counter
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from configurations import Config  # 导入配置类
config = Config()

def load_or_create_vocab(output_folder, train_captions_data, min_word_count):
    vocab_path = os.path.join(output_folder, 'vocab.json')
    if os.path.exists(vocab_path):
        # 如果词汇表已经存在，直接加载
        with open(vocab_path, 'r') as f:
            word_to_idx = json.load(f)
    else:
        # 如果词汇表不存在，创建新的
        vocab = Counter()
        for caption in train_captions_data.values():
            vocab.update(caption.lower().split())
        vocab = {word for word, count in vocab.items() if count >= min_word_count}
        word_to_idx = {word: idx + 4 for idx, word in enumerate(vocab)}
        word_to_idx['<pad>'] = 0
        word_to_idx['<start>'] = 1
        word_to_idx['<end>'] = 2
        word_to_idx['<unk>'] = 3

        # 保存词汇表
        with open(vocab_path, 'w') as f:
            json.dump(word_to_idx, f)
    return word_to_idx

def create_dataset(max_len=64):
    # 检查已存在的文件
    if all(os.path.exists(os.path.join(config.output_folder, fname)) for fname in [
        'encoded_captions_train.json',
        'encoded_captions_test.json',
        'image_paths_train.json',
        'image_paths_test.json',
        'caplens_train.json',
        'caplens_test.json'
    ]):
        print("Dataset already created, skipping creation.")
        return
    image_folder = config.images_path
    train_captions_path = config.train_captions_path
    test_captions_path = config.test_captions_path
    output_folder = config.output_folder

    # 读取图像描述
    with open(train_captions_path, 'r') as f:
        train_captions_data = json.load(f)
    with open(test_captions_path, 'r') as f:
        test_captions_data = json.load(f)

    word_to_idx = load_or_create_vocab(output_folder, train_captions_data, config.min_word_count)


    # 一个函数来转换描述为词索引向量，并进行填充
    def encode_captions(captions_data, word_to_idx, max_len):
        encoded_captions = {}
        caplens = {}
        for img_id, caption in captions_data.items():
            words = caption.lower().split()
            encoded_caption = [word_to_idx.get(word, word_to_idx['<unk>']) for word in words]
            caplen = min(len(encoded_caption) + 2, max_len) - 1
            encoded_caption = [word_to_idx['<start>']] + encoded_caption + [word_to_idx['<end>']]
            encoded_caption += [word_to_idx['<pad>']] * (max_len - len(encoded_caption))
            encoded_captions[img_id] = encoded_caption[:max_len]
            caplens[img_id] = caplen  # if caplen <= max_len else max_len
        return encoded_captions, caplens

    encoded_captions_train, caplens_train = encode_captions(train_captions_data, word_to_idx, max_len)
    encoded_captions_test, caplens_test = encode_captions(test_captions_data, word_to_idx, max_len)

    with open(os.path.join(output_folder, 'vocab.json'), 'w') as f:
        json.dump(word_to_idx, f)
    with open(os.path.join(output_folder, 'encoded_captions_train.json'), 'w') as f:
        json.dump(encoded_captions_train, f)
    with open(os.path.join(output_folder, 'encoded_captions_test.json'), 'w') as f:
        json.dump(encoded_captions_test, f)

    # 存储图像路径
    image_paths_train = {img_id: os.path.join(image_folder, img_id) for img_id in train_captions_data.keys()}
    with open(os.path.join(output_folder, 'image_paths_train.json'), 'w') as f:
        json.dump(image_paths_train, f)

    image_paths_test = {img_id: os.path.join(image_folder, img_id) for img_id in test_captions_data.keys()}
    with open(os.path.join(output_folder, 'image_paths_test.json'), 'w') as f:
        json.dump(image_paths_test, f)
    with open(os.path.join(output_folder, 'caplens_train.json'), 'w') as f:
        json.dump(caplens_train, f)
    with open(os.path.join(output_folder, 'caplens_test.json'), 'w') as f:
        json.dump(caplens_test, f)


# 调用函数，整理数据集
create_dataset()


class ImageTextDataset(Dataset):
    def __init__(self, image_paths_file, captions_file, caplens_file, transform=None):
        """
        参数:
            image_paths_file: 包含图像路径的json文件路径。
            captions_file: 包含编码后文本描述的json文件路径。
            transform: 应用于图像的预处理转换。
        """
        # 载入图像路径和文本描述以及caplens
        with open(image_paths_file, 'r') as f:
            self.image_paths = json.load(f)

        with open(captions_file, 'r') as f:
            self.captions = json.load(f)

        with open(caplens_file, 'r') as f:
            self.caplens = json.load(f)

        # 设置图像预处理方法
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        """
        获取单个数据点。
        参数:
            index: 数据点的索引。
        返回:
            一个包含图像和对应文本描述的元组。
        """
        # 获取图像路径和文本描述以及caplen
        image_id = list(self.image_paths.keys())[index]
        image_path = self.image_paths[image_id]
        caption = self.captions[image_id]
        caplen = self.caplens[image_id]

        # 加载图像并应用预处理
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # 将文本描述转换为张量
        caption_tensor = torch.tensor(caption, dtype=torch.long)

        return image, caption_tensor, caplen,image_id

    def __len__(self):
        """
        数据集中的数据点总数。
        """
        return len(self.image_paths)


# 创建训练集和测试集的 DataLoader
def create_dataloaders(config):
    """
    创建训练集和测试集的 DataLoader。

    参数:
        batch_size: 每个批次的大小。
        num_workers: 加载数据时使用的进程数。
        shuffle_train: 是否打乱训练数据。

    返回:
        train_loader: 训练数据的 DataLoader。
        test_loader: 测试数据的 DataLoader。
    """
    # 图像预处理转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(config.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据时使用的进程数
    num_workers = 0

    # 创建数据集对象
    train_dataset = ImageTextDataset(
        image_paths_file=os.path.join(config.output_folder, 'image_paths_train.json'),
        captions_file=os.path.join(config.output_folder, 'encoded_captions_train.json'),
        caplens_file=os.path.join(config.output_folder, 'caplens_train.json'),
        transform=transform
    )

    test_dataset = ImageTextDataset(
        image_paths_file=os.path.join(config.output_folder, 'image_paths_test.json'),
        captions_file=os.path.join(config.output_folder, 'encoded_captions_test.json'),
        caplens_file=os.path.join(config.output_folder, 'caplens_test.json'),
        transform=transform
    )

    # 创建 DataLoader 对象
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last = True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last = True
    )

    return train_loader, test_loader


config = Config()
# 使用Config类中定义的配置来创建DataLoader
train_loader, test_loader = create_dataloaders(config=config)

def main():
    # 测试 train_loader 和 test_loader 的基本功能
    print("Testing DataLoader with the first two samples:")

    # 获取前两个训练样本
    print("\nFirst two samples from the training data:")
    train_iter = iter(train_loader)
    train_batch = next(train_iter)  # 获取第一个批次
    images, captions, caplens = train_batch

    # 打印训练样本信息
    print(f"Images shape: {images.shape}")  # 图像张量形状
    print(f"Captions: {captions[:2]}")      # 前两条字幕
    print(f"Caption lengths: {caplens[:2]}")  # 前两条字幕的长度


if __name__ == '__main__':
    main()
'''
# 测试 DataLoader 是否正确创建
if __name__ == '__main__':
    for i, (images, captions, caplens) in enumerate(train_loader):
        print(f"Batch {i + 1}")
        print(f"Images shape: {images.size()}")
        print(f"Captions shape: {captions.size()}")
        if i == 1:  # 仅打印前两个批次的信息
            break
'''