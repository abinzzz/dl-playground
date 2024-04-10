import torch

# 替换成你的.bin文件路径
bin_file_path = '/Users/chenyubin/Desktop/no_emo/github/d2l/mnt/combined_weights.bin'

# 加载权重文件
weights = torch.load(bin_file_path)

# 遍历权重字典，打印每个参数的名称和形状
for name, param in weights.items():
    print(f"Parameter name: {name}, Shape: {param.shape}")