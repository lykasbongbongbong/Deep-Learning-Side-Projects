from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader, TensorDataset
# from models import ResNet18, ResNet50
import torch
def resnet_18_w_pretrained():
    batch_size = 256
    epochs = 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #load data
    trainset = RetinopathyLoader(root="data", mode="train")
    testset = RetinopathyLoader(root="data", mode="test")
    train_loader = Dataset(TensorDataset(trainset), batch_size=batch_size, shuffle=True)
    test_loader = Dataset(TensorDataset(testset), batch_size=batch_size, shuffle=False)


    # train_data, train_label =  dataloader.getData("train")
    # test_data, test_label = dataloader.getData("test")
    # print(type(train_label))
    # ads()
    # train_data = torch.from_numpy(train_data)
    # train_label = torch.from_numpy(train_label)
    # test_data = torch.from_numpy(test_data)
    # test_label = torch.from_numpy(test_label)
    # train_loader = DataLoader(TensorDataset(train_data, train_label), batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=batch_size, shuffle=False)


    # asd()
    # model = ResNet18(pretrained=True)
    # model.to(device)
    # model.train()
    # Loss = nn.CrossEntropyLoss()

    # for epoch in range(1, epochs+1):
    #     for _, (data, label) in enumerate(train_loader):
    #         data = data.to(device)
    #         label = label.to(device)
    #         pred = model(data)
    #         loss = Loss(pred, label)
    #         loss.backward()

# def resnet_18_wo_pretrained():

# def resnet_50_w_pretrained():
    
# def resnet_50_wo_pretrained():

if __name__ == '__main__':
    resnet_18_w_pretrained()
    # resnet_18_wo_pretrained()
    # resnet_50_w_pretrained()
    # resnet_50_wo_pretrained()