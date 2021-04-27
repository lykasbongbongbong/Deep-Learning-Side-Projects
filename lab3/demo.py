import dataloader
from torch.optim import Adam, SGD
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import numpy as np
from utils.plot_result import plot_accuracy
from utils.models import EEGNet, DeepConvNet
import os



from dataloader import read_bci_data
def demo():
    epochs = 100
    

    #put things into cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    batch_size = 256
    _, _, test_data, test_label = dataloader.read_bci_data()
    test_data = torch.from_numpy(test_data)
    test_label = torch.from_numpy(test_label)

    test_loader = DataLoader(TensorDataset(test_data, test_label), batch_size = batch_size, shuffle=False)

    
    
    #test EEGNet
    activation = nn.ReLU()
    EEGNet_model = EEGNet(activation)
    EEGNet_model.to(device)
       
    EEGNet_model.load_state_dict(torch.load("weight/EEGNet.weight"))
    EEGNet_model.eval()

    acc = 0.
    for _, (data, label) in enumerate(test_loader):
        data = data.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        pred = EEGNet_model(data)
        acc += pred.max(dim=1)[1].eq(label).sum().item()
    acc = 100. * acc/len(test_loader.dataset)
    print(f"EEGNet Accuracy: {acc:.4f}")




    # #test DeepConvNet
    # if e_activation is "LeakyReLU":
    #     activation = nn.LeakyReLU(negative_slope=0.1)
    # elif e_activation is "ReLU":
    #     activation = nn.ReLU()
    # elif e_activation is "ELU":
    #     activation = nn.ELU()
        
    # EEGNet_model = DeepConvNet(activation)
    # EEGNet_model.to(device)
        
    # EEGNet_model.load_state_dict(torch.load(DeepConvNet))
    # EEGNet_model.eval()

    # acc = 0
    # for _, (data, label) in enumerate(test_loader):
    #     data = data.to(device, dtype=torch.float)
    #     label = label.to(device, dtype=torch.long)
    #     pred = EEGNet_model(data)
    #     acc += pred.max(dim=1)[1].eq(label).sum().item()
    # acc = 100. * acc/len(test_loader.dataset)
    # print(f"DeepConvNet Accuracy: {acc:.4f}")


if __name__ == '__main__':
    demo()