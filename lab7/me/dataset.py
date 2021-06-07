import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import numpy as np


class ICLEVRLoader(data.Dataset):
    def __init__(self, root_folder, data_json_path, trans=None, cond=False, mode='train'):
        self.root_folder = root_folder
        self.num_classes = 24
        self.mode = mode
        self.img_list = []
        self.img_cond = []
        self.data = {}

        with open(os.path.join(root_folder, 'objects.json'), 'r') as file:
            self.objects = json.load(file)
        with open(data_json_path, 'r') as file:
            self.data = json.load(file)
        
        for img_path, img_conditions in self.data.items():
            # CLEVR_train_002066_0.png # ["cyan cube"]
            self.img_list.append(img_path)
            self.img_cond.append([self.objects[cond] for cond in img_conditions])

        self.transformations=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        if self.mode == 'train':
            print("> Found %d images..." % (len(self.img_list)))
                
    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_folder+"/images", self.img_list[index])).convert('RGB')
        img = self.transformations(img)
        condition = self.get_condition_vec(self.img_cond[index])
        return img, condition 
    
    def get_condition_vec(self, int_list):
        one_hot_vec = torch.zeros(self.num_classes)
        for i in int_list:
            one_hot_vec[i]=1.
        return one_hot_vec