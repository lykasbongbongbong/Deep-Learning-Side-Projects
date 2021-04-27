from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
def main():
    #hyper params
    batch_size = 256
    epochs = 1000

    #load training and testing data
    train_data = RetinopathyLoader(root="data", mode="train")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = RetinopathyLoader(root='data', mode='test')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    #train
    # with pretrained weight
    model = ResNet18()

    #test

    

if __name__ == '__main__':
    main()