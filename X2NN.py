import numpy as np
import torch.nn as nn
import torch
from PIL import  Image
import matplotlib.pyplot as plt
import torch.utils.data as data

# 待学习的结构 3*3的卷积网络
Conv=nn.Conv2d(1,1,3,1)
weight=torch.ones_like(Conv.weight.data)
bias=torch.ones_like(Conv.bias.data)
Conv.weight.data=weight
Conv.bias.data=bias

#全连接网络
class x2nn(nn.Module):
    def __init__(self,inputsize,outputsize):
        super(x2nn,self).__init__()
        self.L1=nn.Linear(inputsize,100)
        self.L2=nn.Linear(100,100)
        self.L3=nn.Linear(100,outputsize)
        #self.sig=nn.Sigmoid()
        self.sig=nn.LeakyRelu(0.2)
    def forward(self,x):
        out=self.L1(x)
        out=self.sig(out)
        out=self.L2(out)
        out=self.sig(out)
        out=self.L3(out)
        return out

#实例化模型
mode=x2nn(36,16)
ops=torch.optim.SGD(mode.parameters(),lr=0.001,momentum=0.9,weight_decay=0.1)
loss_f=nn.MSELoss()

#构造数据
class trainset(data.Dataset):
    def __init__(self):
        self.a=torch.rand(1,1,6,6)
        y=Conv(self.a)
        self.y=y.view(-1,1)
    def __getitem__(self,index):
        self.a=torch.rand(1,1,6,6)
        y=Conv(self.a)
        self.y=y.data.view(1,-1)
        return (self.a.view(1,-1),self.y)
    def __len__(self):
        return 100000
train=trainset()
traindata=data.DataLoader(train,batch_size=500,shuffle=True)
cuda=0
mode=mode.cuda(cuda)

#测试数据
test=torch.rand(1,1,6,6)
test=torch.ones_like(test)
test_y=Conv(test)
test_y=test_y.view(1,-1)
mode(test.view(1,-1))
loss_f(mode(test.view(1,-1)),test_y)
mode(test.view(1,-1))

#训练
for j in range(10):
    print(j)
    #测试
    print(mode(test.view(1,-1).cuda(cuda)))
    for i ,k in enumerate(traindata):
        ops.zero_grad()
        out=mode(k[0].cuda(cuda))
        loss=loss_f(out,k[1].cuda(cuda))
        loss.backward()
        ops.step()
        if i%20==0:
            print(loss)
#     print(i)
#     print(k[0])
#     print("*"*20)
#     print(k[1])

#保存模型验证
torch.save(mode,"./mode.pth")
net=torch.load('./mode.pth')
x=torch.rand(1,1,36)
print(net(x.cuda(cuda)))
y=x.view(1,1,6,6)
print(Conv(y))
