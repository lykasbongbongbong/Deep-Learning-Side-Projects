import pandas as pd
from torch.utils import data
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import os


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    elif mode == 'test':
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        raise Exception("Wrong Mode! Type only train/test")

class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): dataset 的 root path
            mode : 要做 training / testing

            self.img_name (string list): 存mode下所有的image (list)
            self.label (int or float list): 存mode下所有image的label (list)
        """
        self.root = root
        self.all_image, self.all_labels = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.all_image)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.all_image)

    def __getitem__(self, index):
        # support the indexing such taht dataset[i] can be used to get i-th sample
        single_img_path = os.path.join(self.root, self.all_image[index]+".jpeg")
        single_image = Image.open(single_img_path)
        single_image_label = self.all_labels[index]
        tran = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ToTensor(), transforms.Normalize((0.3749, 0.2602, 0.1857),(0.2526, 0.1780, 0.1291))])
        single_image_transformed = tran(single_image)

        return single_image_transformed, single_image_label

