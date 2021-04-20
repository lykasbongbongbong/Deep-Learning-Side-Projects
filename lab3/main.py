import dataloader
from torch.optim import Adam
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import matplotlib.pyplot as plt


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet,self).__init__()
        self.firstconv=nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1,1), padding=(0,25), bias=False),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=(2,1), stride=(1,1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1,1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
            nn.Dropout(p=0.25)
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
    
    def forward(self, x):
        out = self.firstconv(x)
        out = self.depthwiseConv(out)
        out = self.separableConv(out)
        out = out.view(out.shape[0], -1)
        out = self.classify(out)
        return out
    
    

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    batch_size = 256
    learning_rate = 0.001
    epochs = 500

    #data: fetch and convert to tensor/put it into dataloader
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label)
    test_data = torch.from_numpy(test_data)
    test_label = torch.from_numpy(test_label)

    train_set = TensorDataset(train_data, train_label)
    test_set = TensorDataset(test_data, test_label)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    

  

    
    #train
    EEGNet_model = EEGNet()
    EEGNet_model.to(device)
    Loss = nn.CrossEntropyLoss()
    optimizer = Adam(EEGNet_model.parameters(), lr=learning_rate, weight_decay=0.01)
    for epoch in range(1, epochs+1):
        EEGNet_model.train()
        total_loss = 0
        accuracy = 0
        for _, (data, label) in enumerate(train_loader):
            data = data.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            pred = EEGNet_model(data)
            loss = Loss(pred, label)
            total_loss += loss.item()
            accuracy += pred.max(dim=1)[1].eq(label).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss /= len(train_loader.dataset)
        accuracy = 100.*accuracy/len(train_loader.dataset)
        if epoch % 10 == 0:
            print(f"epoch:{epoch} loss:{total_loss} accuracy:{accuracy}")





    # model = model.EEGNet()
    # optimizer = Adam(model.parameters(), lr=learning_rate)
    # 
   


    # #檢查有沒有GPU:
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     loss = loss.cuda()
    #     train_data = train_data.cuda()
    #     train_label = train_label.cuda()
    #     test_data = test_data.cuda()
    #     test_label = test_label.cuda()
    
    # accuracy_train = list()
    # accuracy_test = list()
    
    # #train
    # for epoch in range(epochs):
    #     model.train()
    #     train_loss = 0
    #     optimizer.zero_grad()
    #     output_train = model(train_data)
            
    

    

    
    #test
    # print(model)

if __name__ == '__main__':
    main()