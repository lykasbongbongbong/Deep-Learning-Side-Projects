import pandas as pd
import torch
from tqdm import tqdm
def set_parameter_requires_grad(model, mode):
    '''
    mode: feature_extracting / fine_tuning
    '''
    if mode == "feature_extracting":
        for param in model.parameters():
            param.requires_grad = False
    elif mode == "fine_tuning":
        for param in model.parameters():
            param.requires_grad = True


def train_eval(model, train_loader, test_loader, epochs, Loss, optimizer, device, weight_name):
    model.to(device)

    dataframe = pd.DataFrame()
    
    total_train_accuracy = list()
    total_test_accuracy = list()
    best_acc = 0.
    best_model = None
    for epoch in tqdm(range(1, epochs+1)):
        #train
        model.train()
        num_true_per_epoch = 0. 
        loss_per_epoch = 0.

        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            #pred.shape: [8, 5] : batch_size * classes
           
            loss = Loss(pred, labels)
            loss_per_epoch += loss.item()
           
            '''
            torch.return_types.max(
                values=tensor([0.0790, 0.2014, 0.0623, 0.1243, 0.0744, 0.0563, 0.1231, 0.2395],
                    device='cuda:0', grad_fn=<MaxBackward0>),
                indices=tensor([4, 4, 4, 4, 4, 4, 4, 4], device='cuda:0'))
            所以取[1] 會拿到有高分的class
            '''
            # print(pred_class.eq(labels))  #去判斷predict出來的class和labels有沒有一致: return True/False Tensor
            # print(pred_class.eq(labels).sum())  #回傳有幾個true，用.item()拿出tensor中的value, 會是int的型態
            # print(pred_class.eq(labels).sum().item())

            high_prob_class = pred.max(dim=1)  #回傳value/index的tensor
            pred_class = high_prob_class[1]  #取index (class)
            compare = pred_class.eq(labels)  #True/False Tensor
            num_true = compare.sum() #True的總數 Tensor
            num_true_per_epoch += num_true.item()  #t
            loss.backward()
            optimizer.step()
        train_epoch_loss = loss_per_epoch / len(train_loader.dataset)
        train_epoch_acc = 100. * num_true_per_epoch / len(train_loader.dataset)
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
    torch.save(best_model, weight_name)

    dataframe['train_accuracy'] = total_train_accuracy
    dataframe['test_accuracy'] = total_test_accuracy

    return dataframe

    


