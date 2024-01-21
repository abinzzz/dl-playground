import torch
import torch.nn as nn

embedding = nn.Embedding(10, 3)
# 2个样本的批量，每个样本有4个索引
input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
print(input.shape)
print(embedding(input).shape)



embedding = nn.Embedding(10, 3, padding_idx=0)
input = torch.LongTensor([[0, 2, 0, 5]])
embedding(input)
print(embedding(input).shape)
print(input.shape)