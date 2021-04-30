import pandas as pd
import torch
from tqdm import tqdm
def train_eval(model, train_loader, test_loader, epochs, Loss, optimizer, device):
    model.to(device)

    dataframe = pd.DataFrame()
    
    total_train_accuracy = list()
    total_test_accuracy = list()
    best_acc = 0.

    for epoch in tqdm(range(1, epochs+1)):
        #train
        model.train()
        train_batch_acc = 0. 
        total_loss = 0.

        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = Loss(pred, labels)
            total_loss += loss.item()
            train_batch_acc += pred.max(dim=1)[1].eq(labels).sum().item()
            loss.backward()
            optimizer.step()
        train_epoch_loss = total_loss / len(train_loader.dataset)
        train_epoch_acc = 100. * train_batch_acc / len(train_loader.dataset)
        print(f"\nEpoch {epoch} Loss: {train_epoch_loss} Acc: {train_epoch_acc}")
        total_train_accuracy.append(train_epoch_acc)


        #eval
        '''
        model.eval 和 torch.no_grad()有差別
        model.eval會告訴每一層說現在是在evaluate的mode, 所以batchnorm/dropout都不會有功能
        torch.no_grad()則是讓autograd不要運作, 也會加速 
        '''
        with torch.no_grad():
            model.eval()
            test_batch_acc = 0. 
            for images, labels in tqdm(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                pred = model(images)
                test_batch_acc += pred.max(dim=1)[1].eq(labels).sum().item()
            test_epoch_acc = 100. * test_batch_acc / len(test_loader.dataset)
            print(f"\nEpoch {epoch} Acc: {test_epoch_acc}")
            total_test_accuracy.append(test_epoch_acc)
            if test_epoch_acc > best_acc: 
                best_acc = test_epoch_acc
                best_model = model.state_dict()
    torch.save(best_model, "weights/r50_without_pretrained.weight")

    dataframe['train_accuracy'] = total_train_accuracy
    dataframe['test_accuracy'] = total_test_accuracy

    return dataframe

    


