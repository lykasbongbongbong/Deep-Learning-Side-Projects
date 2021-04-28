from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader, TensorDataset
# from models import ResNet18, ResNet50
import torch.nn as nn
from models import ResNet

import torch
def resnet_18_w_pretrained():
    batch_size = 16
    epochs = 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #load data
    # 因為RetinopathyLoader有繼承data.Dataset 所以回傳是一個dataloader object
    trainset = RetinopathyLoader(root="data", mode="train")
    testset = RetinopathyLoader(root="data", mode="test")
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = ResNet(pretrained=True)
    model.to(device)
    model.train()
    Loss = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        for _, (data, label) in enumerate(train_loader):
            '''
            data.shape [256, 3, 512, 512]
            label.shape [256]
            '''
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            loss = Loss(pred, label)
            print(loss.shape)
            asd()
            loss.backward()

# def resnet_18_wo_pretrained():

# def resnet_50_w_pretrained():
    
# def resnet_50_wo_pretrained():

if __name__ == '__main__':
    resnet_18_w_pretrained()
    # resnet_18_wo_pretrained()
    # resnet_50_w_pretrained()
    # resnet_50_wo_pretrained()