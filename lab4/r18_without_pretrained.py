from utils.dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
from models.resnet_models import ResNet18
import torch 
import pandas as pd
from torch.optim import SGD
import torch.nn as nn
from utils.common import train_eval
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                
def resnet_18_without_pretrained():
     #hyper param:
    batch_size = 16
    epochs = 10
    lr = 1e-3
    momentum = 0.9
    Loss = nn.CrossEntropyLoss()
    weight_name = "weights/resnet18_without_pretrained.weight"

    trainset = RetinopathyLoader(root="data", mode="train")
    testset = RetinopathyLoader(root="data", mode="test")
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = ResNet18(classes=5, pretrained=False)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    train_eval_result = train_eval(model, train_loader, test_loader, epochs, Loss, optimizer, device, weight_name)
    train_eval_result.to_csv("result/resnet18_without_pretrained_accuracy.csv", index=False)


if __name__ == '__main__':
    resnet_18_without_pretrained()
