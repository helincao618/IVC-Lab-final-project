import numpy as np
import pandas as pd
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.image as mpimg
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import time
import imageio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm_notebook as tqdm
from PIL import Image

device = 'cpu'

def getdataset(Q):
    featpath2 = './dataset5/compression_ratio_'
    labelpath2 = './dataset5/Groundtruth/'
    train = compset(featpath2+Q,labelpath2)
    return train

def findlabel(featurepath,labelimgs):
    labelroot = './dataset5/Groundtruth/'
    target = re.findall('(\w+[0-9]+\.jpg)',featurepath)
    index = labelimgs.index(labelroot+target[0])
    return index

class compset(Dataset):
    def __init__(self,featroot,labelroot):
        featimgs = os.listdir(featroot)
        self.featimgs=[os.path.join(featroot,k) for k in featimgs]
        labelimgs = os.listdir(labelroot)
        self.labelimgs=[os.path.join(labelroot,k) for k in labelimgs]
        self.transforms = transforms.Compose([transforms.Resize((288,352)),transforms.ToTensor()])    
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
    def __len__(self):
        return len(self.featimgs)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:               #initial COV layer
        nn.init.normal_(m.weight.data, 0.0, 0.02)  
    elif classname.find('BatchNorm') != -1:        #initial BN layer
        nn.init.normal_(m.weight.data, 1.0, 0.02)  
        nn.init.constant_(m.bias.data, 0)          #set bias as constant


IMG_WIDTH = 352
IMG_HEIGHT = 288
num_channels_in_encoder = 3

'Generator Model'
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Generator
        #input size 64*64
        self.e_conv_1 = nn.Sequential(
            nn.Conv2d(3,64,3,2,1) 
        )
        self.e_conv_2 = nn.Sequential( 
            nn.LeakyReLU(),
            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128)
        )
        self.e_conv_3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256)   
        )
        self.e_conv_4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(256,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.g_dconv_1 = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(256)
        )
        self.g_block_1 = nn.Sequential(
            nn.ReLU()
        )   
        self.g_dconv_2 = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),
        )
        self.g_block_2 = nn.Sequential(
            nn.ReLU()
        ) 
        self.g_dconv_3 = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
        ) 
        self.g_block_3 = nn.Sequential(
            nn.ReLU()
        ) 
        self.g_dconv_4 = nn.Sequential(
            nn.ConvTranspose2d(64,3,4,2,1),
        )
        
    def forward(self,x):
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        ec3 = self.e_conv_3(ec2)
        ec4 = self.e_conv_4(ec3)
        dc1 = self.g_dconv_1(ec4)
        dc2 = self.g_block_1(dc1+ec3) #prevent the gradienten loss
        dc2 = self.g_dconv_2(dc2)
        dc2 = self.g_block_2(dc2+ec2)
        dc3 = self.g_dconv_3(dc2)
        dc3 = self.g_block_3(dc3+ec1)
        dc4 = self.g_dconv_4(dc3)
        return dc4 #the output of the G(x)

netG = Generator().to(device)
netG.apply(weights_init)
netG.load_state_dict(torch.load("netG3.model"))
netG.eval()


torch.cuda.empty_cache()

PSNR_test = np.zeros([21,10])
PSNR_original = np.zeros([21,10])
print(PSNR_test.shape)

j = 0
print('starting testing loop')
Q = ['1.140', '2.004', '2.908', '3.399','4.061', '4.523','5.248', '6.298', '8.387', '13.352']
for q in Q:
    dataset = getdataset(q)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True)
    i = 0
    for i, (feature,label) in enumerate(dataloader, 0):
            netG.eval()
            real_images = label.to(device)    
            compressed_img = feature.to(device)
            fake_images = netG(compressed_img) 
            
            fake_images = fake_images.squeeze(0)
            real_images = real_images.squeeze(0)
            compressed_img = compressed_img.squeeze(0)
            
            fake_images = fake_images.detach().numpy()
            real_images = real_images.detach().numpy()
            compressed_img = compressed_img.detach().numpy()
            
            mse_1 = np.sum((real_images-fake_images)**2)/304128
            PSNR_test[i][j] = 10*np.log10(1/(mse_1))
            
            mse_2 = np.sum((real_images-compressed_img)**2)/304128
            PSNR_original[i][j] = 10*np.log10(1/(mse_2))
            
            min_val = np.min(fake_images)
            max_val = np.max(fake_images)
            fake_images = (fake_images - min_val) / (max_val - min_val)
            plt.imsave(str(i)+str(j)+'.png',np.transpose(fake_images,(1,2,0)))
            i = i+1
    j = j+1 
    
print(np.mean(PSNR_test,0))
print(np.mean(PSNR_original,0))

