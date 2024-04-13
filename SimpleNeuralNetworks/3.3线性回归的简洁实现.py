
#1.生成数据集
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=d2l.synthetic_data(true_w,true_b,1000)

# print(features)
# print(labels)

#2.读取数据集
def load_array(data_arrays,batch_size,is_train=True):  #@save
    """构造一个pytorch数据迭代器"""
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size=10
data_iter=load_array((features,labels),batch_size)

print(next(iter(data_iter))) #从iter迭代器中取出下一个元素，next取出一个x,一个y

#3.定义模型

# nn是神经网络的缩写
from torch import nn

#Sequential可以理解成list of layers
net = nn.Sequential(nn.Linear(2, 1))

#4.初始化模型参数
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

#5.定义损失函数
loss = nn.MSELoss() #均方误差

#6.定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

#7.训练
num_epochs=3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y) #计算小批量loss，方便进行梯度后向传播更新参数
        trainer.zero_grad()#梯度累计清零
        l.backward()
        trainer.step()
    l=loss(net(features),labels)#计算整个数据集的loss
    print(f'epoch{epoch+1},loss{l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)