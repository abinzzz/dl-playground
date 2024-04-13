#%matplotlib inline
import random
import torch
from d2l import torch as d2l

#1.生成数据集
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))# （均值，方差，行数，列数）
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)#加入噪音
    return X, y.reshape((-1, 1))
## y.reshape((-1, 1)),这里的-1大小会自动计算，这里把y做成了列向量

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)




print('features:', features[0],'\nlabel:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);##（横轴，纵轴，点的大小）

import matplotlib.pyplot as plt
plt.show()




## 2.读取数据集
def data_iter(batch_size,feature,labels):
    num_examples=len(features) #样本总数
    indices =list(range(num_examples)) #索引列表

    #样本是随机读取的
    random.shuffle(indices)#索引列表被打乱

    for i in range(0,num_examples,batch_size):

        #批次索引[i:i+batchsize]或[i:num_examples]
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])

        #yield 关键字用于返回当前批次的数据。
        #它使 data_iter 函数成为一个生成器。
        #每次调用这个生成器时，它都会返回数据集中的下一个批次，直到遍历完整个数据集。
        yield features[batch_indices],labels[batch_indices]


#读取第一个小批量数据样本并打印
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

#3.初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)#追踪于这两个参数的梯度计算，使用反向传播更新参数
b = torch.zeros(1, requires_grad=True)

##4.定义模型
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

##5.定义损失函数
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

##6.定义优化算法
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():#更新的时候不要参与梯度计算
        for param in params:
            param -= lr * param.grad / batch_size #更新参数
            param.grad.zero_() #清0梯度积累，使得下一次计算梯度的时候，不和上次相关

## 7.训练
lr = 0.03
num_epochs = 3#把整个数据扫三遍
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels): #遍历整个训练数据集
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward() #反向传播计算损失
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
