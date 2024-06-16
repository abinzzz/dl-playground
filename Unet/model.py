import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Moudle):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

class Unet(nn.Moudle):
    def __init__(self,in_channel=3,out_channel=1,features=[64,128,256,512]):
        super(Unet,self).__init__()

        self.ups=nn.MoudleList()
        self.downs=nn.MoudleList()
        self.pools=nn.Maxpool2d(kernel_size=2,stride=2)

        for feature in features:
            self.downs.append(D)