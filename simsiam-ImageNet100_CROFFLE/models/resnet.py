import torch
import torch.nn as nn
from torchvision.models import *
from collections import OrderedDict
import torch.nn.functional as F
import re
import numpy as np

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class ResNet(nn.Module):
    def __init__(self,resnet='resnet50'):
        super(ResNet,  self).__init__()
        if resnet == 'resnet50':
            self.resnet = resnet50(pretrained=False)
        elif resnet == 'resnet101':
            self.resnet = resnet101(pretrained=False)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.conv2 = self.resnet.layer1
        self.conv3 = self.resnet.layer2
        self.conv4 = self.resnet.layer3
        self.conv5 = self.resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = self.resnet.fc.in_features
        
        # self.conv2norm = nn.InstanceNorm2d(256)
        # self.conv3norm = nn.InstanceNorm2d(512)
        # self.conv4norm = nn.InstanceNorm2d(1024)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # bottom-up pathway
        C1 = self.conv1(x)
        C2 = self.conv2(C1) # stride of 4  (224 / 4 = 56) C2.shape=[batch, 256, 56, 56] 
        C3 = self.conv3(C2) # stride of 8  (224 / 8 = 28) C3.shape=[batch, 512, 28 ,28]
        C4 = self.conv4(C3) # stride of 16 (224 / 16 = 14) C4.shape=[batch, 1024, 14, 14]
        C5 = self.conv5(C4) # stride of 32 (224 / 32 = 7) C5.shape=[batch, 2048, 28, 28]

        # print('C2:',C2.shape) # [64, 256, 7, 7]
        # print('C3:',C3.shape) # [64, 512, 7, 7]
        # print('C4:',C4.shape) # [64, 1024, 7, 7] 
        # C2 = self.avgpool(C2)
        # C2 = torch.flatten(C2, 1)

        # C3 = self.avgpool(C3)
        # C3 = torch.flatten(C3, 1)

        # C4 = self.avgpool(C4)
        # C4 = torch.flatten(C4, 1)

        out = self.avgpool(C5)
        out = torch.flatten(out, 1)
        # print('out:',out.shape)

        return C2, C3, C4, C5, out

