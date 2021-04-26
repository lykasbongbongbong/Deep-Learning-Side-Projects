import dataloader
import torch 
import torch.nn as nn
from torch.optim import Adam 
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import numpy as np
import os
from utils.plot_result import plot_accuracy
from utils.models import DeepConvNet


            

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #init params
    epochs = 300
    batch_size = 256
    learning_rate = 0.001
    Loss = nn.CrossEntropyLoss()
    print_interval = 10

    #data: fetch and convert to tensor/put it into dataloader
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label)
    test_data = torch.from_numpy(test_data)
    test_label = torch.from_numpy(test_label)

    train_loader = DataLoader(TensorDataset(train_data, train_label), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=batch_size, shuffle=False)

    # activation_list = ["LeakyReLU", "ReLU", "ELU"]
    activation_list = ["LeakyReLU"]



    train_acc_dict = {"LeakyReLU":[], "ReLU":[], "ELU":[]}
    test_acc_dict = {"LeakyReLU":[], "ReLU":[], "ELU":[]}

    best_accuracy = 0.
    best_training_accuracy = 0.
    

    for activate_func in activation_list:
        if activate_func is "LeakyReLU":
            activation = nn.LeakyReLU(negative_slope=0.1)
        elif activate_func is "ReLU":
            activation = nn.ReLU()
        elif activate_func is "ELU":
            activation = nn.ELU()
        
        model = DeepConvNet(activation)
        model.to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

        print(f"\n\n-----Activation Function: {activation}-----")
        
        for epoch in range(1, epochs+1):
            if epoch % print_interval == 0:
                print(f"---Epoch: {epoch}---")

            #train
            model.train()
            total_loss = 0
            acc = 0.
            for _, (data, label) in enumerate(train_loader):
                data = data.to(device, dtype=torch.float )
                label = label.to(device, dtype=torch.long)
                y_pred = model(data)
                loss = Loss(y_pred, label)
                total_loss += loss.item()
                acc += y_pred.max(dim=1)[1].eq(label).sum().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss = total_loss/len(train_loader.dataset)
            acc = 100.*acc/len(train_loader.dataset) 
            if acc > best_training_accuracy:
                best_training_accuracy = acc
            train_acc_dict[activate_func].append(acc)
            
            if epoch % print_interval == 0:
                print(f"[Training] loss:{total_loss:.4f} accuracy:{acc:.1f}")

            #test
            model.eval()
            total_loss = 0
            acc = 0.
            for _, (data, label) in enumerate(test_loader):
                data = data.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.long)
                y_pred = model(data)
                loss = Loss(y_pred, label)
                total_loss += loss.item()
                acc += y_pred.max(dim=1)[1].eq(label).sum().item()
            total_loss = total_loss/len(test_loader.dataset)
            acc = 100.*acc/len(test_loader.dataset)
            test_acc_dict[activate_func].append(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_activation = activate_func
                best_model = model.state_dict()
            if epoch % print_interval == 0:
                print(f"[Testing] loss:{total_loss:.4f} accuracy:{acc:.1f}")

    # print(f"Best Training Accuracy: {best_training_accuracy}")
    print(f"Best Testing Accuracy: {best_accuracy}")
    print(f"Best Activation Function: {best_activation}")

        
    weight_name = "weight/DeepConvNet.weight"
    folder = weight_name.split('/')[0]
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(best_model, weight_name)
    
    plot_accuracy(epochs, train_acc_dict, test_acc_dict, "DeepConvNet", "result/DeepConvNet.png")





if __name__ == '__main__':
    main()