from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader 
from resnet_models import ResNet50
import torch.nn as nn
from torch.optim import SGD
from utils import train_eval
def r50_without_pretrained():
    #hyper param: 
    batch_size = 16 
    classes = 5 
    epochs = 10
    Loss = nn.CrossEntropyLoss()
    lr = 1e-4
    weight_decay = 5e-4    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    #load data
    trainset = RetinopathyLoader(root="data", mode="train")
    testset = RetinopathyLoader(root="data", mode="test")
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    #model
    model = ResNet50(classes=classes, pretrained=False)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    model.to(device)

    train_test(model, train_loader, test_loader, epochs, Loss, optimizer)
    
if __name__ == '__main__':
    r50_without_pretrained()