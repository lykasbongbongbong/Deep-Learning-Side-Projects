from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
from resnet_models import ResNet18
import torch 
import pandas as pd
from torch.optim import Adam, SGD
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def train_eval(model, train_loader, test_loader, epochs, optimizer):
    batch_size = 64
    lr = 1e-3
    momentum = 0.9
    Loss = nn.CrossEntropyLoss()

    dataframe = pd.DataFrame()
    best_accuracy = 0. 
    best_weight = dict()

    model.to(device)
    train_accuracy = list()
    test_accuracy = list() 
    for epoch in range(1, epochs+1):
        #train
        print(f"---Training---")
        with torch.set_grad_enabled(True):
            model.train()
            total_loss = 0
            cor = 0 
            acc = 0.
            for images, labels in train_loader:
                optimizer.zero_grad()
                images = images.to(device)
                labels = labels.to(device)
                pred = model(images)
                loss = Loss(pred, labels)
                total_loss += loss.item()
                cor += pred.max(dim=1)[1].eq(labels).sum().item()
                loss.backward()
                optimizer.step() 
            total_loss /= len(train_loader.dataset)
            acc = 100. * cor / len(train_loader.dataset)
            train_accuracy.append(acc)
            print(f"[Training] Epoch {epoch} loss:{total_loss:.4f} acc:{acc:.4f}")
        #eval
        with torch.set_grad_enabled(False):
            print(f"---Evaluating---")
            model.eval()
            cor = 0
            acc = 0.
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                pred = model(images)
                cor += pred.max(dim=1)[1].eq(labels).sum().item()
            acc = 100. * cor/len(test_loader.dataset)
            test_accuracy.append(acc)
            if acc > best_accuracy:
                best_accuracy = acc 
                best_model = model.state_dict()
            print(f"[Testing] Epoch {epoch}  acc:{acc:.4f}")
    dataframe['train_accuracy'] = train_accuracy
    dataframe['test_accuracy'] = test_accuracy
    print(f"---Saving Model's weight---")
    torch.save(best_model, "weights/resnet18_with_pretrained.weight")

    return dataframe
                
def resnet_18_with_pretrained():
    #hyper param:
    batch_size = 64
    lr = 1e-3
    momentum = 0.9
    Loss = nn.CrossEntropyLoss()
    epochs_feature_extraction = 1
    epochs_fine_tuning = 1
    weight_path = "weights/resnet18_with_pretrained.weight"

    trainset = RetinopathyLoader(root="data", mode="train")
    testset = RetinopathyLoader(root="data", mode="test")
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = ResNet18(classes=5 ,pretrained=True)
    # model.load_state_dict(torch.load(weight_path))
    print(f"---Model loads pretrained weight from {weight_path}---")
    
    #feature extraction
    print(f"---Start feature extraction for {epochs_feature_extraction} epochs.---")
    params_to_update = list()
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    optimizer = SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=5e-4)
    dataframe_feature_extraction = train_eval(model, train_loader, test_loader, epochs_feature_extraction, optimizer)

    #fine tune
    print(f"---Start fine tuning for {epochs_fine_tuning} epochs.---")
    for param in model.parameters():
        param.requires_grad=True
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    dataframe_fine_tuning = train_eval(model, train_loader, test_loader, epochs_fine_tuning, optimizer)
    
    dataframe_ff = pd.concat([dataframe_feature_extraction, dataframe_fine_tuning], axis=0, ignore_index=True)
    print(dataframe_ff)
def resnet_18_without_pretrained():
     #hyper param:
    batch_size = 64
    epochs = 2
    lr = 1e-3
    momentum = 0.9
    Loss = nn.CrossEntropyLoss()

    trainset = RetinopathyLoader(root="data", mode="train")
    testset = RetinopathyLoader(root="data", mode="test")
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = ResNet18(classes=5 ,pretrained=True)
    # model.load_state_dict(torch.load("models/resnet18_with_pretraining_81.pt"))
    model.to(device)
    
    dataframe = pd.DataFrame()

    model = ResNet18(classes=5, pretrained=False)
    model.to(device)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)

    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        train_acc = 0. 
        total_loss = 0. 
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = Loss(pred, labels)
            total_loss += loss.item()
            loss.backward()
            train_acc += pred.max(dim=1)[1].eq(labels).sum().item()
            optimizer.step()
        total_loss /= len(train_loader.dataset)
        train_acc = 100.*train_acc/len(train_loader.dataset)
        print(f"[Training]Epoch: {epoch} loss:{total_loss}, acc: {train_acc}")
        
        model.eval()
        test_acc = 0.
        total_loss = 0. 
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            test_acc += pred.max(dim=1)[1].eq(labels).sum().item()
        total_loss /= len(test_loader.dataset)
        test_acc = 100.*test_acc/len(test_loader.dataset)
        if test_acc > best_acc:
            best_acc = test_acc 
            best_model = model.state_dict()
        print(f"[Testing]Epoch: {epoch} acc: {test_acc}")
    dataframe['acc_train'] = train_acc
    dataframe['acc_test'] = test_acc

    print("---ResNet18 without pretrained weight---")
    print(dataframe)
    torch.save(best_weight, "weights/resnet18_without_pretrained.weight")
           

if __name__ == '__main__':
    
    resnet_18_with_pretrained()
    # resnet_18_without_pretrained()
