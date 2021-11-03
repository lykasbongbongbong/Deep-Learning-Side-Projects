import torch 
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, activation=nn.ReLU()):
        super(EEGNet,self).__init__()
        self.activation = activation
        self.firstconv=nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1,1), padding=(0,25), bias=False),
           nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=(2,1), stride=(1,1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation, 
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1,1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation, 
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
            nn.Dropout(p=0.25)
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
    
    def forward(self, x):
        out = self.firstconv(x)
        # print("[Batch_size, feature_map_size, feature_map_width, feature_map_height]")
        # print(out.shape)
        out = self.depthwiseConv(out)
        out = self.separableConv(out)
        out = out.view(out.shape[0], -1)
        out = self.classify(out)
        return out


class DeepConvNet(nn.Module):
    def __init__(self, activation=nn.ReLU()):
        super(DeepConvNet,self).__init__()
        self.C = 2
        self.T = 750
        self.N = 2
        self.activation = activation
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1,5), stride=(1,1), bias=False),
            nn.Conv2d(25, 25, kernel_size = (self.C, 1), stride=(1,1), bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size = (1, 5), stride=(1,1), bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size = (1, 5), stride=(1,1), bias=False),
            # nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size = (1, 5), stride=(1,1), bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5)
        )

        self.classify = nn.Sequential(
            nn.Linear(in_features=8600, out_features=2, bias=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.shape[0], -1)
        out = self.classify(out)
        return out

