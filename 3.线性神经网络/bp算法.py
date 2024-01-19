import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim):
        super(Net,self).__init__()
        self.layer1=nn.Linear(in_dim,hid_dim)
        self.layer2=nn.Linear(hid_dim,out_dim)

    def forward(self,x):
        x=nn.ReLU(self.layer1(x))
        x=self.layer2(x)
        return x

net=Net(2,4,1)

loss=nn.MSELoss()
opt=torch.optim.SGD(net.parameters(),lr=0.01)

x=torch.randn(100,2)
y=2*x[:,0]+3*x[:,1]+0.5


for epoch in range(100):
    pred=net(x)
    loss1=loss(pred,y)

    opt.zero_grad()
    loss1.backward()
    opt.step()

mdoel=model.cuda()
device_ids=[0,1]
model=torch.nn.DataParallel(model,device_ids)

