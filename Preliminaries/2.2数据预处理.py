
#1.读取数据集
print("----------1.读取数据集----------")
#人工创建数据集
import os
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

#加载数据集
import pandas as pd
data=pd.read_csv(data_file)
print(data)

print("----------2.处理缺失值----------")
#2.处理缺失值
#“NaN”项代表缺失值。 为了处理缺失的数据，典型的方法包括插值法和删除法， 其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。

#插值法
inputs,outputs=data.iloc[:,0:2],data.iloc[:,2]
print(inputs)

inputs=inputs.fillna(inputs.mean())#用inputs的平均值去填充缺失值
print(inputs)

inputs=pd.get_dummies(inputs,dummy_na=True)#one-hot编码？
print(inputs)


##3.numpy转换为张量格式
print("----------3.转换为张量格式----------")
import torch
X=torch.tensor(inputs.to_numpy(dtype=float))
y=torch.tensor(outputs.to_numpy(dtype=float))
print(type(X))

#练习1
print("----------练习1----------")
print(data)
names=["NumRooms","Alley","Price"]
#isnull()函数默认axis=0

# print("有无缺失值：",data.isnull())
# print("统计缺失值数量：",data.isnull().sum())
max_n=data.isnull().sum().max() #统计最大缺失值的数量
#print(data.isnull().sum())
#def del_col(data):
for n in names:
    if data[n].isnull().sum()==max_n:
        data_drop=data.drop(columns=n)

print(data_drop)

#最优答案
drop_data = data.dropna(axis=1, thresh=data.count().min()+1) #有最少实值的数量+！，就删去了那个含有最小实值的列
print(drop_data)
#print(drop_data)
#练习2
print("----------练习2----------")
print(data)
data1=pd.get_dummies(data,dummy_na=True)
print(data1)
print(data1.to_numpy())

print(torch.tensor(data1.to_numpy()))
