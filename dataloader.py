import numpy as np 
import os
import re
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def findlabel(featurepath,labelimgs):
    labelroot = './dataset/Groundtruth/'
    target = re.findall('(\w+[0-9]+\.jpg)',featurepath)
    index = labelimgs.index(labelroot+target[0])
    return index

class ImageData(Dataset):
    def __init__(self,featroot,labelroot):
        featimgs = os.listdir(featroot)
        self.featimgs=[os.path.join(featroot,k) for k in featimgs]
        labelimgs = os.listdir(labelroot)
        self.labelimgs=[os.path.join(labelroot,k) for k in labelimgs]
        self.transforms = transforms.Compose([transforms.Resize((288,352)),transforms.ToTensor()])
    
    def __len__(self):
        return len(self.featimgs)
    
    def __getitem__(self, index):
        feature_img_path = self.featimgs[index]
        feature_img = Image.open(feature_img_path)
        if self.transforms:
            feature = self.transforms(feature_img)
        else:
            feature = np.asarray(feature_img)
            feature = torch.from_numpy(feature) 
        labelindex = findlabel(feature_img_path,self.labelimgs);
        label_img_path = self.labelimgs[labelindex]
        label_img = Image.open(label_img_path)
        if self.transforms:
            label = self.transforms(label_img)
        else:
            label = np.asarray(label_img)
            label = torch.from_numpy(label)
        data = (feature,label)
        return data