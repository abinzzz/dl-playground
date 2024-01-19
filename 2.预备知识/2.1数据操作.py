
## 1.入门
print("----------1.入门----------")
import torch
x=torch.arange(12)
print(x)
print("张量的形状:",x.shape)#张量的形状
print("张量中元素总数:",x.numel())#张量中元素总数

X=x.reshape(3,4) #转换形状
print("转换形状好的x：",X)

x0=torch.zeros((2,3,4))
print("全0张量：",x0)

x1=torch.ones((2,3,4))
print("全1张量：",x1)

x2=torch.randn(3,4)
print("正态张量：",x2)

x3=torch.tensor([[1,2,3],[4,6,7]])
print("自定义张量：",x3)

## 2.运算符
print("----------2.运算符----------")
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print("x:",x)
print("y:",y)
print("x+y:",x+y)
print("x-y:",x-y)
print("x*y:",x*y)
print("x/y:",x/y)
print("指数运算：",torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("X:",X)
print("Y:",Y)
print(torch.cat((X, Y), dim=0)) ##dim表示在第几层维度进行合并(即第几层中括号)
print(torch.cat((X, Y), dim=1))
print("每个位置是否相等：",X==Y)
print("张量中所有元素求和：",X.sum())

## 3.广播机制
print("----------3.广播机制----------")
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print("a:",a)
print("b:",b)
print("a+b:",a+b)

## 4.索引和切片
print("----------4.索引和切片----------")
print("X:",X)
print("X的最后一个元素：",X[-1])
print("X的第二个和第三个元素,即(1,3]",X[1:3])

X[1,2]=9 #指定元素
X[0:2,:]=12 #指定行

## 5.节省内存
print("----------5.节省内存----------")
##   X+=Y是原地操作
##   X=X+Y是新建对象

before=id(Y)
Y=Y+X
print(id(id)==before)

#使用X[:]来减少内存开销
before1=id(X)
X[:]=X+Y
print(id(X)==before1)

## 6.转换为其他python对象

#ndarray和tensor的互换 A➡️B
A=X.numpy()
B=torch.tensor(A)
print("A:",type(A))
print("B:",type(B))

#要将大小为1的张量转换为Python标量，我们可以调用item函数或Python的内置函数。
a=torch.tensor([3.5])
print(type(a))
print(type(float(a)))

print("----------练习1----------")
print("每个位置是否满足条件：",X>Y)

print("----------练习2----------")#条件是对应的位置两者必须是倍数关系，因为是倍数关系，才能广播过去
a1=torch.arange(12).reshape(2,3,2)
a2=torch.arange(6).reshape(2,3,1)
print("a1+a2:",a1+a2)