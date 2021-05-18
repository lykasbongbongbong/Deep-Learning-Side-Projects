import seaborn as sns 
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
import itertools
import csv
import numpy as np
import torch 
from utils.dataloader import RetinopathyLoader
from models.resnet_models import ResNet50, ResNet18
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import reduce 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

 



def demo(y_pred):
    #ground truth
    y_true = list()
    with open("test_label.csv", "r") as file:
        y_true = file.read().splitlines()
    y_true = y_true[1:]
    
    # #pred source 
    # y_pred = list()
    # with open("confusion_matrix_source/resnet50_with_pretrained_pred_class.csv", "r") as file:
    #     y_pred = file.read().splitlines()
    
    # y_pred = list(map(int, y_pred))
    y_true = list(map(int, y_true))


    labels = [0,1,2,3,4]
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    plt.figure()
    plot_confusion_matrix(cm, classes=["0", "1", "2", "3", "4"], normalize=True, title="ResNet50 with pretrained")
    plt.savefig("2021_confusion_matrix_res50_with_pretrained.png")

def evaluate():
    batch_size = 16

    #load data
    testset = RetinopathyLoader(root="data", mode="test")
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    #testing
    model = ResNet50(classes=5, pretrained=True)
    model.load_state_dict(torch.load("weights/resnet50_with_pretraining.pt"))
    model.to(device)

    model.eval()
    correct = 0

    y_pred = list()
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images)
        pred_class = pred.max(dim=1)[1]
        y_pred.append(pred_class.tolist())
        correct+=pred.max(dim=1)[1].eq(labels).sum().item()

       
    acc=100.*correct/len(test_loader.dataset)
    print(f"Evaluation Acc: {acc:.4f}")
    

    y_pred = reduce(lambda x,y :x+y ,y_pred)
    return y_pred
   

if __name__ == '__main__':
    y_pred = evaluate()
    print(confusion_matrix)
    demo(y_pred)