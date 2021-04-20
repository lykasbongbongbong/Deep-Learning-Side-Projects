import dataloader
import train
from torch.optim import Adam
import torch.nn as nn
import torch

def main():
    batch_size = 64
    learning_rate = 1e-2
    epochs = 300

    
    #處理data
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    #train
    model = train.EEGNet()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()


    #檢查有沒有GPU:
    if torch.cuda.is_available():
        print("here!")
        model = model.cuda()
        loss = loss.cuda()
    
    #test
    print(model)

if __name__ == '__main__':
    main()