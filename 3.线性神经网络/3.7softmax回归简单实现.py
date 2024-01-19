import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#1.初始化模型参数

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net=nn.Sequential(nn.Flatten(),nn.Linear(784,10)) #展平层和线性层
def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)

#2.重新审视softmax的实现
loss = nn.CrossEntropyLoss(reduction='none')

#3.优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# net=nn.Sequential()
# net.add_moudle(nn.Conv2d(3,3,3))
# net.add_moudle(nn.BatchNorm2d(3))
# net.add_module(nn.ReLU())
#
# net=nn.Sequential(nn.Conv2d(3,3,3,),nn.BatchNorm2d(3),nn.ReLU())


params=model.parameters()

opt=optim.SGD(params,lr=0.1)

opt.zero_grad()
loss.backward()
opt.step()

torch.nn.parallel.data_parallel(moudle,inputs,device_id=None)\

model=model.cuda()
device_ids=[0,1]
model=torch.nn.DataParallel(model,device_ids=device_ids)