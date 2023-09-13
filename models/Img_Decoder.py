import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

class ImageDecoder(nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=4,stride=2,padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=8,stride=4,padding=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128,out_channels=32,kernel_size=16,stride=8,padding=4)
    
        self.fusion = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=1)
        self.fusion_bn = torch.nn.BatchNorm2d(32)

    def forward(self, I0, I1, I2):
        Deconv = []

        d1 = self.deconv1(I0)
        Deconv.append(d1)

        d2 = self.deconv2(I1)
        Deconv.append(d2)

        d3 = self.deconv3(I2)
        Deconv.append(d3)

        de_concat = torch.cat(Deconv,dim=1)
        img_fusion = F.relu(self.fusion_bn(self.fusion(de_concat)))

        return img_fusion

if __name__ == "__main__":
    data1 = torch.zeros(size=(32,64,80,60))
    data2 = torch.zeros(size=(32,64,40,30))
    data3 = torch.zeros(size=(32,128,20,15))

    ie = ImageDecoder()
    result = ie(data1,data2,data3)

    print(result.shape)

