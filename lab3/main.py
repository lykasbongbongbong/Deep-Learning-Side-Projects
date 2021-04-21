import dataloader
from torch.optim import Adam, SGD
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import numpy as np


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
        out = self.depthwiseConv(out)
        out = self.separableConv(out)
        out = out.view(out.shape[0], -1)
        out = self.classify(out)
        return out



def main(pretrained_train=False, pretrained_test=False, do_train=False, do_test=False, save_model=False, pretrained_path="EEGNet.weight"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    batch_size = 256
    learning_rate = 0.001
    epochs = 300
    activation_list = ["LeakyReLU", "ReLU", "ELU"]
    print_interval = 100

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
        optimizer = Adam(EEGNet_model.parameters(), lr=learning_rate, weight_decay=0.001)
        # optimizer = SGD(EEGNet_model.parameters(), lr=learning_rate, momentum=0.9)
        print(f"\n\n-----Activation Function: {activation}-----")
        
        for epoch in range(1,epochs+1):
            if epoch % print_interval == 0:
                print(f"epoch:{epoch}")
            
            #train
            if do_train:
                if pretrained_train:
                    EEGNet_model.load_state_dict(torch.load(pretrained_path))
                EEGNet_model.train()
                total_loss = 0
                acc = 0
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
                acc_train_dict[activate_func].append(acc)
                if epoch % print_interval == 0:
                    print(f"[Training] loss:{total_loss:.4f} accuracy:{acc:.1f}")

            #test
            if do_test:
                if pretrained_test:
                    EEGNet_model.load_state_dict(torch.load(pretrained_path))
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
                    best_model_state = EEGNet_model.state_dict()
                    best_activation = activate_func
                if epoch % print_interval == 0:
                    print(f"[Testing] loss:{total_loss:.4f} accuracy:{acc:.1f}")
               

    print(f"Best Activation: {best_activation}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    if save_model:
        torch.save(best_model_state, "EGGNet.weight")
    plot_result(epochs, acc_train_dict, acc_test_dict)

def plot_result(epochs, acc_train_dict, acc_test_dict):
    # fig, ax = plt.subplots()  # Create a figure and an axes.
    
    x = np.arange(1, epochs+1)
    plt.plot(x, acc_train_dict["ReLU"], label='relu_train')  # Plot some data on the axes.
    plt.plot(x, acc_test_dict["ReLU"], label='relu_test')  # Plot some data on the axes.
    plt.plot(x, acc_train_dict["LeakyReLU"], label='leaky_relu_train')  # Plot more data on the axes...
    plt.plot(x, acc_test_dict["LeakyReLU"], label='leaky_relu_test')  # ... and some more.
    plt.plot(x, acc_train_dict["ELU"], label='elu_train')  # Plot more data on the axes...
    plt.plot(x, acc_test_dict["ELU"], label='elu_test')  # ... and some more.
    plt.ylabel('Accuracy(%)')  # Add an x-label to the axes.
    plt.xlabel('Epoch')  # Add a y-label to the axes.
    plt.title("Activation function comparison(EEGNet)")  # Add a title to the axes.
    plt.legend()  # Add a legend.
    plt.savefig('EGGNet.png')


if __name__ == '__main__':
    do_train = True 
    do_test = True
    pretrained_train = False
    pretrained_test = False
    save_model = False
    pretrained_path = "EEGNet.weight"
    main(pretrained_train, pretrained_test, do_train, do_test, save_model, pretrained_path)