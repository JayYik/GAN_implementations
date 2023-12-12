import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from PIL import Image
from sklearn.model_selection import train_test_split
class CelebA(Dataset):
    def __init__(self, root,split,transform=None,resolution=64):
        if split == 'test':
            self.train=False
        else:
            self.train=True
        self.root=self.get_root_byresolution(root,resolution)
        self.transform = transform
        img_names = [i for i in os.listdir(self.root) if i.endswith(".jpg")]
        self.img_list = [os.path.join(self.root, i) for i in img_names]
        self.train_img_list ,self.test_img_list = train_test_split(self.img_list,test_size=0.2,random_state=114514)


    def __getitem__(self, index):
        if self.train:
            img = Image.open(self.train_img_list[index]).convert('RGB')
        else:
            img = Image.open(self.test_img_list[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img,torch.tensor(0)

    def __len__(self):
        if self.train:
            return len(self.train_img_list)
        else:
            return len(self.test_img_list)
        
    def get_root_byresolution(self,root,resolution):
        data_dir=os.path.join(root,'celeba')
        if resolution==64:
            data_dir=os.path.join(data_dir,'celeba-64')
        elif resolution==128:
            data_dir=os.path.join(data_dir,'celeba-128')
        elif resolution==256:
            data_dir=os.path.join(data_dir,'celeba-256')
        elif resolution==512:
            data_dir=os.path.join(data_dir,'celeba-512')
        elif resolution==1024:
            data_dir=os.path.join(data_dir,'celeba-1024')
        return data_dir