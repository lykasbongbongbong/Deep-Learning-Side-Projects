from utils.dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
from models.resnet_models import ResNet18
import torch 
import pandas as pd
from torch.optim import SGD
import torch.nn as nn
from utils.common import train_eval
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def resnet_18_with_pretrained():
    #hyper param:
    batch_size = 64
    lr = 1e-3
    momentum = 0.9
    Loss = nn.CrossEntropyLoss()
    epochs_feature_extraction = 5
    epochs_fine_tuning = 5
    weight_name = "weights/resnet18_with_pretrained.weight"

    trainset = RetinopathyLoader(root="data", mode="train")
    testset = RetinopathyLoader(root="data", mode="test")
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = ResNet18(classes=5 ,pretrained=True)
    
    
    #feature extraction
    print(f"---Start feature extraction for {epochs_feature_extraction} epochs.---")
    params_to_update = list()
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    optimizer = SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=5e-4)
    train_eval_result_fe = train_eval(model, train_loader, test_loader, epochs_feature_extraction, Loss, optimizer, device, weight_name)
    train_eval_result_fe.to_csv("result/r18_with_pretrained_accuracy_feature_extraction.csv", index=False)
    print(train_eval_result_fe)

    #fine tune
    print(f"---Start fine tuning for {epochs_fine_tuning} epochs.---")
    for param in model.parameters():
        param.requires_grad=True
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    train_eval_result_ft = train_eval(model, train_loader, test_loader, epochs_fine_tuning, Loss, optimizer, device, weight_name)
    train_eval_result_ft.to_csv("result/r18_with_pretrained_accuracy_fine_tuning.csv", index=False)
    print(train_eval_result_ft)

    
    frames = [train_eval_result_fe, train_eval_result_ft]
    train_eval_result = pd.concat(frames)
    print(train_eval_result)
    train_eval_result.to_csv("result/resnet18_with_pretrained_accuracy.csv", index=False)

if __name__ == '__main__':
    resnet_18_with_pretrained()
