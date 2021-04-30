from utils.dataloader import RetinopathyLoader
from resnet_models import ResNet50
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch
from torch.optim import SGD  
from utils.common import set_parameter_requires_grad, train_eval
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def r50_with_pretained():
    #hyper params
    batch_size = 16
    classes = 5
    Loss = nn.CrossEntropyLoss()
    lr = 5e-4
    momentum = 0.9
    weight_decay = 1e-3
    epochs_feature_extraction = 10
    epochs_fine_tune = 30
    weight_name = "weights/resnet50_with_pretrained.weight"


    #load data
    trainset = RetinopathyLoader(root="data", mode="train")
    testset = RetinopathyLoader(root="data", mode="test")
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False) 

    #model 
    model = ResNet50(classes=classes, pretrained=True)
    
    # 一般而言 load一個pretrained model之後所以的參數都會被設成requires_grad=True
    # 但如果是要做feature extracting (只要算新initialize的layer的gradient 那其他的parameter就不要requires_grad)
    # set_parameter_requires_grad(model, "feature_extracting")
    param_to_update = []
    for layer_name, layer_param in model.named_parameters():
        #每一層的weight: (layer_name, parameters)
        if layer_param.requires_grad:
            #print(layer_name) 只有fc
            param_to_update.append(layer_param) 
    
    optimizer = SGD(param_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)
    train_eval_result_fe = train_eval(model, train_loader, test_loader, epochs_feature_extraction, Loss, optimizer, device, weight_name)
    print(train_eval_result_fe)

    set_parameter_requires_grad(model, "fine_tuning")
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    train_eval_result_ft = train_eval(model, train_loader, test_loader, epochs_fine_tune, Loss, optimizer, device, weight_name)
    print(train_eval_result_ft)

    frames = [train_eval_result_fe, train_eval_result_ft]
    train_eval_result = pd.concat(frames)
    print(train_eval_result)

    train_eval_result.to_csv("result/r50_with_pretrained.csv")



if __name__ == '__main__':
    r50_with_pretained()