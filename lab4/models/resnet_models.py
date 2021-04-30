import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import collections
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.cuda.comm as comm
from torch.autograd.function import once_differentiable
import time
import functools

class ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet, self).__init__()

        self.classify = nn.Linear(2048, 5)

        pretrained_model = models.__dict__['resnet{}'.format(50)](pretrained=False)
        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        del pretrained_model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x

class ResNet18(nn.Module):
    def __init__(self, classes, pretrained=False):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        #如果是用已經pretrained過的weight, 參數就不要更新
        if pretrained == True:
            for param in self.model.parameters():
                param.requires_grad = False
        in_features = self.model.fc.in_features
        #最後輸出5個class
        self.model.fc = nn.Linear(in_features, classes)
    
    def forward(self, x):
        out = self.model(x)
        return out


class ResNet50(nn.Module):
    def __init__(self, classes, pretrained=False):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        if pretrained == True:
            for param in self.model.parameters():
                param.requires_grad = False 
        in_features = self.model.fc.in_features 
        self.model.fc = nn.Linear(in_features, classes)
    
    def forward(self, x):
        out = self.model(x)
        return out 