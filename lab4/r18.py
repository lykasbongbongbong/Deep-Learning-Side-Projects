from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
from resnet_models import ResNet18
import torch 
import pandas as pd
from torch.optim import Adam, SGD
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def resnet_18_with_pretrained():
    #hyper param:
    batch_size = 64
    fe_epoch = 5
    ft_epoch = 5
    lr = 1e-3
    momentum = 0.9
    Loss = nn.CrossEntropyLoss()

    trainset = RetinopathyLoader(root="data", mode="train")
    testset = RetinopathyLoader(root="data", mode="test")
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = ResNet18(classes=5 ,pretrained=True)
    model.load_state_dict(torch.load("models/resnet18_with_pretraining_81.pt"))
    model.to(device)
    
    dataframe = pd.DataFrame()

    #先train最後fc (做feature extraction)
    parameters = list()
    for name, param in model.named_parameters():
        if param.requires_grad:
            parameters.append(param)
    optimizer = SGD(parameters, lr=lr, momentum=momentum, weight_decay=5e-4)
    
    model.to(device)
    accuracy_train = list()
    accuracy_test = list()
    best_testing_acc = 0.


    for epoch in range(1, fe_epoch+1):
        #train
        model.train()
        train_acc = 0.
        total_loss = 0.
        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = Loss(pred, labels)
            total_loss += loss.item()
            train_acc+=pred.max(dim=1)[1].eq(labels).sum().item()
            loss.backward()
            optimizer.step()
        total_loss /= len(train_loader.dataset)
        train_acc = 100. * train_acc/len(train_loader.dataset)
        accuracy_train.append(train_acc)
        print(f"[Training] Epoch {epoch}: loss:{total_loss:4f}, acc:{train_acc:.4f}")

        #test
        model.eval()
        test_acc = 0.
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            test_acc += pred.max(dim=1)[1].eq(labels).sum().item()
        test_acc = 100. * test_acc/len(test_loader.dataset)
        if test_acc > best_testing_acc:
            best_testing_acc = test_acc
            best_weight = model.state_dict()
        accuracy_test.append(test_acc)
        print(f"[Testing] Epoch {epoch}: acc:{test_acc:.4f}")
    torch.save(best_weight, "weights/resnet18_pretrained.weight")
    
    
    #做fine tuning (更新參數)
    for param in model.parameters():
        param.require_grad = True
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    model.to(device)
    train_acc = 0. 
    test_acc = 0. 
    best_testing_acc = 0. 
    for epoch in range(1, ft_epoch+1):
        model.train()
        total_loss = 0.
        for images, label in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            label = label.to(device)
            pred = model(images)
            loss = Loss(pred, label)
            total_loss += loss.item()
            train_acc+=pred.max(dim=1)[1].eq(label).sum().item()
            loss.backward()
            optimizer.step()
        total_loss /= len(train_loader.dataset)
        train_acc = 100. * train_acc/len(train_loader.dataset)
        accuracy_train.append(train_acc)
        print(f"[Training] Epoch {epoch}: loss:{total_loss:4f}, acc:{train_acc:.4f}")
        
        #test
        model.eval()
        test_acc = 0.
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            test_acc += pred.max(dim=1)[1].eq(labels).sum().item()
        test_acc = 100. * test_acc/len(test_loader.dataset)
        if test_acc > best_testing_acc:
            best_testing_acc = test_acc
            best_weight = model.state_dict()
        accuracy_test.append(test_acc)
        print(f"[Testing] Epoch {epoch}: acc:{test_acc:.4f}")
    
    dataframe['acc_train'] = accuracy_train
    dataframe['acc_test'] = accuracy_test
    print(dataframe)
    torch.save(best_weight, "weights/resnet18_pretrained.weight")
           
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
    model.load_state_dict(torch.load("models/resnet18_with_pretraining_81.pt"))
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
    # resnet_18_with_pretrained()
    resnet_18_without_pretrained()
