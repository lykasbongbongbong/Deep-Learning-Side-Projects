import dataloader
import torch 
import torch.nn as nn
from torch.optim import Adam 
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import numpy as np


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
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
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

    activation_list = ["LeakyReLU", "ReLU", "ELU"]

    train_acc_dict = {"LeakyReLU":[], "ReLU":[], "ELU":[]}
    test_acc_dict = {"LeakyReLU":[], "ReLU":[], "ELU":[]}

    best_accuracy = 0
    

    for activate_func in activation_list:
        if activate_func is "LeakyReLU":
            activation = nn.LeakyReLU(negative_slope=0.1)
        elif activate_func is "ReLU":
            activation = nn.ReLU()
        elif activate_func is "ELU":
            activation = nn.ELU()
        
        model = DeepConvNet(activation)
        model.to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

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

    print(f"Best Accuracy: {best_accuracy}")
    print(f"Best Activation Function: {best_activation}")

        
    torch.save(best_model, "DeepConvNet.weight")
    plot_result(epochs, train_acc_dict, test_acc_dict)


def plot_result(epochs, train_acc_dict, test_acc_dict):
    # fig, ax = plt.subplots()  # Create a figure and an axes.
    
    x = np.arange(1, epochs+1)
    plt.plot(x, train_acc_dict["ReLU"], label='relu_train')  # Plot some data on the axes.
    plt.plot(x, test_acc_dict["ReLU"], label='relu_test')  # Plot some data on the axes.
    plt.plot(x, train_acc_dict["LeakyReLU"], label='leaky_relu_train')  # Plot more data on the axes...
    plt.plot(x, test_acc_dict["LeakyReLU"], label='leaky_relu_test')  # ... and some more.
    plt.plot(x, train_acc_dict["ELU"], label='elu_train')  # Plot more data on the axes...
    plt.plot(x, test_acc_dict["ELU"], label='elu_test')  # ... and some more.
    plt.ylabel('Accuracy(%)')  # Add an x-label to the axes.
    plt.xlabel('Epoch')  # Add a y-label to the axes.
    plt.title("Activation function comparison (DeepConvNet)")  # Add a title to the axes.
    plt.legend()  # Add a legend.
    plt.savefig('DeepConvNet.png')



if __name__ == '__main__':
    main()