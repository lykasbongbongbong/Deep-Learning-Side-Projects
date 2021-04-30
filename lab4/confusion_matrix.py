from utils.dataloader import RetinopathyLoader 
from torch.utils.data import DataLoader 
from models.resnet_models import ResNet50, ResNet18
import numpy as np
import torch 
import matplotlib.pyplot as plt
from functools import reduce 
from tqdm import tqdm 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def evaluate():
    batch_size = 16

    #load data
    testset = RetinopathyLoader(root="data", mode="test")
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    #testing
    model = ResNet18(classes=5, pretrained=True)
    model.load_state_dict(torch.load("weights/resnet18_without_pretrained.weight"))
    model.to(device)

    model.eval()
    correct = 0
    confusion_matrix=np.zeros((5,5))

    y_pred = list()
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images)
        pred_class = pred.max(dim=1)[1]
        y_pred.append(pred_class.tolist())
        correct+=pred.max(dim=1)[1].eq(labels).sum().item()

        for i in range(len(labels)):
            confusion_matrix[int(labels[i])][int(pred_class[i])] += 1
    acc=100.*correct/len(test_loader.dataset)
    print(f"Acc: {acc}")
    # normalize confusion_matrix
    confusion_matrix=confusion_matrix/confusion_matrix.sum(axis=1).reshape(5,1)

    y_pred = reduce(lambda x,y :x+y ,y_pred)
    print(len(y_pred))
   
    with open("confusion_matrix_source/resnet18_without_pretrained_pred_class.csv", "w") as file:
        for ele in y_pred:
            file.write(str(ele)+"\n")
    
    return confusion_matrix

# def plot_confusion_matrix(confusion_matrix):
#     print("---plotting confusion matrix---")
#     fig, ax = plt.subplots(figsize=(6,6))
#     ax.xaxis.set_label.position('top')
#     for i in range(confusion_matrix.shape[0]):
#         print(i)
#         for j in range(confusion_matrix.shape[1]):
#             print(j)
#             ax.text(i, j, '{:.2f}'.format(confusion_matrix[j, i]), va='center', ha='center')
#     ax.set_xlabel('Predicted label')
#     ax.set_ylabel('True label')
#     print("---all set---")
#     return fig

if __name__ == '__main__':
    confusion_matrix = evaluate()
    print(confusion_matrix)

    # fig = plot_confusion_matrix(confusion_matrix)
    # print("---saving figure---")
    # fig.savefig("ResNet18_pretrained.png")