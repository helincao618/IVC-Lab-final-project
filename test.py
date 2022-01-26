import numpy as np
import re
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataloader import ImageData
from model import Generator

device = 'cpu'

def getdataset(Q):
    featpath2 = './dataset/compression_ratio_'
    labelpath2 = './dataset/Groundtruth/'
    train = ImageData(featpath2+Q,labelpath2)
    return train

def findlabel(featurepath,labelimgs):
    labelroot = './dataset/Groundtruth/'
    target = re.findall('(\w+[0-9]+\.jpg)',featurepath)
    index = labelimgs.index(labelroot+target[0])
    return index

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:               #initial COV layer
        nn.init.normal_(m.weight.data, 0.0, 0.02)  
    elif classname.find('BatchNorm') != -1:        #initial BN layer
        nn.init.normal_(m.weight.data, 1.0, 0.02)  
        nn.init.constant_(m.bias.data, 0)          #set bias as constant

netG = Generator().to(device)
netG.apply(weights_init)
netG.load_state_dict(torch.load("netG3.model"))
netG.eval()
torch.cuda.empty_cache()
PSNR_test = np.zeros([21,10])
PSNR_original = np.zeros([21,10])

print('starting testing loop')
Q = ['1.140', '2.004', '2.908', '3.399','4.061', '4.523','5.248', '6.298', '8.387', '13.352']
for j, q in enumerate(Q):
    dataset = getdataset(q)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True)
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
        # plt.imsave(str(i)+str(j)+'.png',np.transpose(fake_images,(1,2,0)))
    
print(np.mean(PSNR_test,0))
print(np.mean(PSNR_original,0))
