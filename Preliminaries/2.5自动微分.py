#1.一个简单的例子
import torch

x = torch.arange(4.0)
print(x)

#x的梯度
x.requires_grad=True
print(x.grad)#表示x的梯度

#计算y
y=2*torch.dot(x,x)
print(y) #输出结果28=2(1x1 + 2x2 + 3x3 + 4x4)

y.backward()
print(x.grad) #梯度为4x

print(x.grad==4*x)#验证x梯度为4x

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_() #梯度清0
y = x.sum()
print(y)

y.backward()
print(x.grad)


#2.非标量变量的反向传播

# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)

#3.分离计算？？？ 看不懂
x.grad.zero_()
y = x * x
u = y.detach() #这里u被看作常数，不参与求导
z = u * x

z.sum().backward()
print(x.grad == u)

#4. python控制流的梯度计算

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

print(a.grad == d / a)


