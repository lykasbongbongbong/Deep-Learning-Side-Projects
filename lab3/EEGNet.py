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
from utils.models import EEGNet
import os

def main(pretrained_train=False, save_model=False, pretrained_path="EEGNet.weight"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    batch_size = 256
    learning_rate = 0.001
    epochs = 300
    activation_list = ["LeakyReLU", "ReLU", "ELU"]
    # activation_list = ["ELU"]
    print_interval = 10

    #data: fetch and convert to tensor/put it into dataloader
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label)
    test_data = torch.from_numpy(test_data)
    test_label = torch.from_numpy(test_label)

    train_loader = DataLoader(TensorDataset(train_data, train_label), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=batch_size, shuffle=False)
    
   
    best_activation = 1
    best_accuracy = 0.
    best_model_state = dict()
    acc_train_dict = {"LeakyReLU":[], "ReLU":[], "ELU":[]}
    acc_test_dict = {"LeakyReLU":[], "ReLU":[], "ELU":[]}

    #分開來看best acc 依據不同的AF
    
    best_training_accuracy = 0.
    

    #train
    for activate_func in activation_list:
        if activate_func is "LeakyReLU":
            activation = nn.LeakyReLU(negative_slope=0.1)
        elif activate_func is "ReLU":
            activation = nn.ReLU()
        elif activate_func is "ELU":
            activation = nn.ELU()

        EEGNet_model = EEGNet(activation)
        EEGNet_model.to(device)
        Loss = nn.CrossEntropyLoss()
        optimizer = Adam(EEGNet_model.parameters(), lr=learning_rate, weight_decay=0.01)
        print(f"\n\n-----Activation Function: {activation}-----")
        
        for epoch in range(1,epochs+1):
            if epoch % print_interval == 0:
                print(f"epoch:{epoch}")
            
           
            if pretrained_train:
                EEGNet_model.load_state_dict(torch.load(pretrained_path))
            EEGNet_model.train()
            total_loss = 0
            acc = 0.
            for _, (data, label) in enumerate(train_loader):
                data = data.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.long)
                y_pred = EEGNet_model.forward(data)
                loss = Loss(y_pred, label)
                total_loss += loss.item()
                acc += y_pred.max(dim=1)[1].eq(label).sum().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss /= len(train_loader.dataset)
            acc = 100.*acc/len(train_loader.dataset)
            if acc > best_training_accuracy:
                best_training_accuracy = acc
            acc_train_dict[activate_func].append(acc)
            if epoch % print_interval == 0:
                print(f"[Training] loss:{total_loss:.4f} accuracy:{acc:.1f}")

            #test
           
            EEGNet_model.eval()
            total_loss = 0
            acc = 0
            for _, (data, label) in enumerate(test_loader):
                data = data.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.long)
                pred = EEGNet_model(data)
                loss = Loss(pred, label)
                total_loss += loss.item()
                acc += pred.max(dim=1)[1].eq(label).sum().item()
            total_loss /= len(test_loader.dataset)
            acc = 100. * acc/len(test_loader.dataset)
            acc_test_dict[activate_func].append(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_model_state = copy.deepcopy(EEGNet_model.state_dict())
                best_activation = activate_func
            if epoch % print_interval == 0:
                print(f"[Testing] loss:{total_loss:.4f} accuracy:{acc:.1f}")
               

    print(f"\n\nBest Activation: {best_activation}")
    # print(f"Best training accuracy: {best_training_accuracy:.4f}")
    print(f"Best testing Accuracy: {best_accuracy:.4f}\n\n")
    # print(acc_train_dict)
   
    




    if save_model:
        weight_name = "weight/EEGNet.weight"
        # weight_name = "weight/EEGNet.weight"
        folder = weight_name.split('/')[0]
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(best_model_state, weight_name)
    plot_accuracy(epochs, acc_train_dict, acc_test_dict, "EEGNet", "result/EEGNet.png")



if __name__ == '__main__':
    pretrained_train = False
    save_model = True
    pretrained_path = "weight/EEGNet.weight"
    main(pretrained_train, save_model, pretrained_path)