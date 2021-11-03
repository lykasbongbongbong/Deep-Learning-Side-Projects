
# coding: utf-8

# In[11]:


import seaborn as sns 
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
import itertools
import csv
import numpy as np


# In[18]:


y_true = list()

with open("test_label.csv", "r") as file:
    y_true = file.read().splitlines()


y_true = y_true[1:]


# In[19]:


y_pred = list()

with open("confusion_matrix_source/resnet50_with_pretrained_pred_class.csv", "r") as file:
    y_pred = file.read().splitlines()





# In[20]:


y_pred = list(map(int, y_pred))
y_true = list(map(int, y_true))


# In[21]:


labels = [0,1,2,3,4]
cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# print('Normalized confusion matrix')
# print(cm_normalized)


# In[22]:


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
    


# In[23]:


plt.figure()
plot_confusion_matrix(cm, classes=["0", "1", "2", "3", "4"], normalize=True, title="ResNet50 with pretrained")
plt.savefig("confusion_matrix_res50_pretrained.png")
