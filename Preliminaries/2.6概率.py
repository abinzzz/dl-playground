#1. 基本概率论

#导入必要的软件包
#%matplotlib inline
import matplotlib.pyplot as plt
import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6
a=multinomial.Multinomial(1, fair_probs).sample() #多项分布中投掷骰子一次
print(a) #打印采样结果

b=multinomial.Multinomial(10, fair_probs).sample() #投掷骰子10次
print(b)

# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
print(counts / 1000)  # 相对频率作为估计值 进行1000次的骰子投掷，最终结果接机你与0.167

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
plt.show()


