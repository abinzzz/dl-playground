import torch
from torch import nn

# 创建一个随机输入张量
input_tensor = torch.randn(2, 3, 4, 4)

# 创建归一化层实例
batch_norm = nn.BatchNorm2d(3)
layer_norm = nn.LayerNorm(input_tensor.size()[1:])
instance_norm = nn.InstanceNorm2d(3)
group_norm = nn.GroupNorm(1, 3)  # 将3个通道放入单一组，类似于LayerNorm

# 应用归一化层
output_batch_norm = batch_norm(input_tensor)
output_layer_norm = layer_norm(input_tensor)
output_instance_norm = instance_norm(input_tensor)
output_group_norm = group_norm(input_tensor)

# 打印输出的统计数据
def print_stats(name, tensor):
    print(f"{name}: Mean: {tensor.mean().item():.4f}, Std Dev: {tensor.std().item():.4f}")

print_stats("BatchNorm Output", output_batch_norm)
print_stats("LayerNorm Output", output_layer_norm)
print_stats("InstanceNorm Output", output_instance_norm)
print_stats("GroupNorm Output", output_group_norm)
