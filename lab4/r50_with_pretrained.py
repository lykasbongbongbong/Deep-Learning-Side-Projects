from utils.dataloader import RetinopathyLoader
from resnet_model import ResNet50
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch
from torch.optim import SGD  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def r50_with_pretrained():
    #hyper params
    batch_size = 16
    classes = 5
    Loss = nn.CrossEntropyLoss()
    lr = 5e-4
    momentum = 0.9
    weight_decay = 1e-4
    epochs_feature_extraction = 5
    epochs_fine_tune = 5
    epoch_classifier = 10

    #load data
    trainset = RetinopathyLoader(root="data", mode="train")
    testset = RetinopathyLoader(root="data", mode="test")
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False) 

    #model 
    model = ResNet50(classes=classes, pretrained=True)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    #直接當成classifier
    train_eval = train_eval(model, train_loader, test_loader, epoch_classifier, Loss, optimizer, device)




if __name__ == '__main__':
    r50_with_prertained()