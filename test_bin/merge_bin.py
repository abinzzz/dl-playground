import torch

# # 创建模拟的权重字典
# weights_part1 = {'layer1.weight': torch.randn(5, 5), 'layer2.weight': torch.randn(5, 5)}
# weights_part2 = {'layer1.bias': torch.randn(5), 'layer2.bias': torch.randn(5)}

# # 保存这两个权重字典为两个.bin文件
# torch.save(weights_part1, 'mnt/weights_part1.bin')
# torch.save(weights_part2, 'mnt/weights_part2.bin')


# 加载之前保存的两个权重文件
weights_part1 = torch.load('mnt/weights_part1.bin')
weights_part2 = torch.load('mnt/weights_part2.bin')

# 合并权重字典
combined_weights = {**weights_part1, **weights_part2}

# 合并权重字典
combined_weights = {**weights_part1, **weights_part2}

# 保存合并后的权重为一个新的.bin文件
combined_path = 'mnt/combined_weights.bin'
torch.save(combined_weights, combined_path)