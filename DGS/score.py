import leran
import torch
import random


MAX_LENGTHS=120
embed_size = 300
hidden_size = 256
vocab_size = len(leran.vocab)
BATCH_SIZE = 20
max_length = leran.max_length
EPOCHS = 1
testdata = leran.testdata
test_dataset = leran.test_dataset
test_loader = leran.test_loader
model = leran.ImageCaptioningModel(embed_size, hidden_size, vocab_size, max_length)

# 加载保存的状态字典
model_state_dict = torch.load('model_state_dict.pth')

# 将状态字典应用到模型
model.load_state_dict(model_state_dict)

model.eval()
image_path = random.choice(list(testdata.keys()))
leran.evaluate(image_path)