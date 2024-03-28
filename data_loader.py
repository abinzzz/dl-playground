import torch
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # 返回数据和标签
        return self.data[index]

    def __len__(self):
        # 返回数据集的长度
        return len(self.data)

# 创建数据集
data = [1, 2, 3, 4, 5]
dataset = MyDataset(data)

# 创建数据加载器
batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 遍历数据加载器
for batch in dataloader:
    # 在这里进行训练或推理操作
    print(batch)