print("----------1.标量----------")
#1.标量
import torch
x=torch.tensor(3.0)
y=torch.tensor(2.0)

print("x+y:",x+y)
print("x*y:",x*y)
print("x**y",x**y)#幂次

print("----------2.向量----------")
#2.向量
x=torch.arange(4)
print(x)
print(x[3])

print("----------3.长度，维度和形状----------")
#3.长度，维度和形状

#tensor能使用的语法
print(len(x))
print(x.shape)

print("----------4.矩阵----------")
#4.矩阵
A = torch.arange(20).reshape(5, 4)
print("矩阵A:",A)
print("矩阵A的转置:",A.T)

#5.张量
print("----------5.张量----------")
X=torch.arange(24).reshape(2,3,4)
print(X)

#6.张量算法的基本性质
print("----------6.张量算法的基本性质----------")
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print("A:",A)
print("A*B",A*B) #这是元素对应相乘，而不是矩阵乘法
print("A+2",A+2) #每个元素加上1

# 7.降维
print("----------7.降维----------")
x=torch.arange(4,dtype=torch.float32)
print("x:",x)
print("x.sum:",x.sum())#所有元素求和

print("A.shape:",A.shape)#shapen不是函数，所以不需要加括号
print("A.sum():",A.sum())

A_sum_axis0 = A.sum(axis=0) #要的是横向的，所以把纵向的求和
print("A_sum_axis0:",A_sum_axis0)
print("A_sum_axis0.shape:",A_sum_axis0.shape)


print("平均值：",A.mean())
print("数量：",A.numel())

print("----------8.非降维求和----------")
# 8.非降维求和
sum_A = A.sum(axis=1, keepdims=True) #第二个参数表示保持数组的二维形状
print(sum_A)
print("纵向求和：",A/sum_A)

print(A.cumsum(axis=0)) #计算累计总和

#9.点积
print("----------9.点积----------")
y=torch.ones(4,dtype=torch.float32)
print(x)
print(y)
print("x dot y:",torch.dot(x,y))

#10.向量积
print("----------10.向量积----------")
print(A.shape)#5，4
print(x.shape)#4
print("A和x的向量积:",torch.mv(A,x)) # m v=matrix vector

#11.矩阵乘法，矩阵乘法是向量积的扩展
print("----------11.矩阵乘法---------")
B = torch.ones(4, 3)
print(torch.mm(A,B)) #m m=matrix matirx

#12.范数 l1:元素绝对值之和   l2:元素平方和开方
print("----------12.范数---------")
u = torch.tensor([3.0, -4.0])
print("l2范数：",torch.norm(u))
print("l1范数：",torch.abs(u).sum())


#练习1
print("----------练习1---------")
A=torch.arange(9).reshape(3,3)
print(A.T.T==A)

#练习2
print("----------练习2---------")
B=torch.ones(9).reshape(3,3)
print(A.T+B.T==(A+B).T)

#练习3
#跳过

#练习4
X=torch.arange(24).reshape(2,3,4)
print(len(X))#长度为第一维度

#练习5
Y=torch.arange(10).reshape(5,2)
print(len(Y))

Z=torch.ones(6).reshape(1,3,2)
print(len(Z))

#练习6
A=torch.arange(12).reshape(3,4) #
#print(A/(A.sum(axis=1).T))
print(A)
print(A.sum(axis=1).reshape(3,1))#这个虽然是纵向求和，但是在格式里面的分布是横向的\

print(A/A.sum(axis=1).reshape(3,1))

#练习7 哪个维度求和，哪个维度就消失
Q=torch.arange(24).reshape(2,3,4)
print(Q)
print(Q.sum(axis=0).shape)
print(Q.sum(axis=1).shape)
print(Q.sum(axis=2).shape)

#练习8
import numpy as np
print(np.linalg.norm(Q))#标量