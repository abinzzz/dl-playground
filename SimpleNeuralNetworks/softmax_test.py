def softmax(X):
    X_exp=torch.exp(X)
    print("X_exp",X_exp.shape,X_exp)
    """
    X_exp torch.Size([2, 5]) 
          tensor([[0.6880, 0.8210, 2.1784, 1.2788, 4.2303],
        [1.2565, 0.1625, 0.7192, 0.3345, 0.6678]])
    """
    partition=X_exp.sum(1,keepdim=True) #对行求和保留列
    partition1=X_exp.sum(0,keepdim=True) #对列求和保留行
    print("partition",partition.shape,partition)
    print("partition1",partition1.shape,partition1)
    """
    partition torch.Size([2, 1]) 
              tensor([[9.1965],
        [3.1404]])

    partition1 torch.Size([1, 5]) 
               tensor([[1.9445, 0.9834, 2.8976, 1.6133, 4.8981]])
    """
    print("X_exp/partition",(X_exp/partition).shape,X_exp/partition)
    """
    X_exp/partition torch.Size([2, 5]) 
                    tensor([[0.0748, 0.0893, 0.2369, 0.1391, 0.4600],
        [0.4001, 0.0517, 0.2290, 0.1065, 0.2126]])
    """
    return X_exp/partition


import torch




# 初始化权重W和偏置b
W = torch.normal(0, 1, (5, 10))
b = torch.zeros(10)


X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X)
print(X_prob, X_prob.sum(1))

