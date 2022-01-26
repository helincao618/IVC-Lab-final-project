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
import cv2 as cv
import imageio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from IPython.display import HTML
from tqdm import tqdm_notebook as tqdm
from PIL import Image

device = 'cpu'
img_dir = 'dataset2/'
imagePaths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
img_list = os.listdir(img_dir)
valid_ratio = 0.8
batch_size=2

def psnr(original,constrast):
    mse = np.mean((original-constrast)**2)
    if mse == 0:
        return 100
    PSNR = 10*np.log10(1/(mse))
    return PSNR

"picture pre-produce"
class ImageData(Dataset):
    def __init__(self,is_train=True):
        self.is_train = is_train
        self.transform = transforms.Compose([transforms.ToTensor(),]) # range [0, 255] -> [0.0,1.0]
        self.train_index = int(valid_ratio * len(img_list)) 
        
        self.crop = transforms.CenterCrop((288,352))  #Crop from center
        
    def __len__(self):
        if self.is_train:
            return self.train_index
        else:
            return len(img_list) - self.train_index -1
    def __getitem__(self, index):
        if not self.is_train:
            index = self.train_index + index
#         print("hey  "*4 + str(index))
        img = mpimg.imread(img_dir+img_list[index]) # matplotlib.image as mpimg,Used to read pictures, and read out is the array format
        img = self.crop(TF.to_pil_image(img))       #Convert between PIL image and PyOpenCV matrix
        img = self.transform(img)                   
        img = (img-0.5) /0.5                        #norm

        return img

def getdataset(Q):
    featpath2 = './foreman/compression_ratio_'
    labelpath2 = './foreman/Groundtruth/'
    train = compset(featpath2+Q,labelpath2)
    return train
def findlabel(featurepath,labelimgs):
    labelroot = './foreman/Groundtruth/'
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

Q = ['1.140', '2.004', '2.908', '3.399','4.061', '4.523','5.248', '6.298', '8.387', '13.352']

dataset = getdataset(Q[0])
x,y = dataset[0]
x = x*0.5+0.5
x = np.transpose(x.numpy(),(1,2,0))


dataset = ImageData()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:               #initial cov layer
        nn.init.normal_(m.weight.data, 0.0, 0.02)  
    elif classname.find('BatchNorm') != -1:        #initial BN layer
        nn.init.normal_(m.weight.data, 1.0, 0.02)  
        nn.init.constant_(m.bias.data, 0)          #setting bias to constant

IMG_WIDTH = 288
IMG_HEIGHT = 352
#IMG_WIDTH = 1920
#IMG_HEIGHT = 1072
num_channels_in_encoder = 3

'Encoder model'
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        #conv1
        self.e_conv_1 = nn.Conv2d(3,64,3,2,1)
        #Le+Cov2+BN
        self.e_conv_2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(64,128,4,2,1), #in channels,outputs channels,kernel size,stride,padding,
            nn.BatchNorm2d(128)
        )
        #Le+Cov3+BN
        self.e_conv_3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256)  
        )
        #Le+Cov4+BN+Relu
        self.e_conv_4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(256,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
    def forward(self,x):
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        ec3 = self.e_conv_3(ec2)
        ec4 = self.e_conv_4(ec3)
        return ec4

netE = Encoder().to(device)
netE.apply(weights_init)

inp = torch.randn(IMG_WIDTH*IMG_HEIGHT*3 * 1)
inp = inp.view((-1,3,IMG_HEIGHT,IMG_WIDTH))

input1 = inp.to(device)
output = netE(input1)

'Generator Model'
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #Conv1
        self.e_conv_1 = nn.Sequential(
            nn.Conv2d(3,64,3,2,1) #in_channels, output channels, kernel size,stride,padding,
        )
        #Le+Cov2+BN
        self.e_conv_2 = nn.Sequential( 
            nn.LeakyReLU(),
            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128)
        )
        #Le+Cov3+BN
        self.e_conv_3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256)   
        )
        #Le+Cov4+BN+Relu ec4
        self.e_conv_4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(256,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        #Dcov1+BN
        self.g_dconv_1 = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(256)
        )
        #Relu
        self.g_block_1 = nn.Sequential(
            nn.ReLU()
        )   
       #Dcov2+BN
        self.g_dconv_2 = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1),
            #nn.ZeroPad2d((1, 1, 1, 1)),
            nn.BatchNorm2d(128),
        )
        #Relu
        self.g_block_2 = nn.Sequential(
            nn.ReLU()
        ) 
        #Dcov3+BN
        self.g_dconv_3 = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
        )
        #Relu    
        self.g_block_3 = nn.Sequential(
            nn.ReLU()
        ) 
        #Dcov4
        self.g_dconv_4 = nn.Sequential(
            nn.ConvTranspose2d(64,3,4,2,1),
        )
        
    def forward(self,x):
        ec1 = self.e_conv_1(x)
        print(ec1.shape)
        
        ec2 = self.e_conv_2(ec1)
        print(ec2.shape)
        
        ec3 = self.e_conv_3(ec2)
        print(ec3.shape)
        
        ec4 = self.e_conv_4(ec3)
        print(ec4.shape)
        
        dc1 = self.g_dconv_1(ec4)
        print(dc1.shape)
        
        dc2 = self.g_block_1(dc1+ec3) #prevent the gradienten loss
        print(dc2.shape)
        
        dc2 = self.g_dconv_2(dc2)
        print(dc2.shape)
        
        dc2 = self.g_block_2(dc2+ec2)
        print(dc2.shape)
        
        dc3 = self.g_dconv_3(dc2)
        print(dc3.shape)
        
        dc3 = self.g_block_3(dc3+ec1)
        print(dc3.shape)
        
        dc4 = self.g_dconv_4(dc3)
        print(dc4.shape)
        
        return dc4 #the output of the G(x)

netG = Generator().to(device)
netG.apply(weights_init)

inp = torch.randn(2*num_channels_in_encoder*IMG_WIDTH*IMG_HEIGHT)
inp = inp.view((-1,3,IMG_HEIGHT,IMG_WIDTH))
inp = inp.to(device)
print(inp.shape)

output = netG(inp)
print(output.shape)

torch.cuda.empty_cache()

output = output.clamp(0.0, 1.0)
output = output.cpu() 


'Discriminato Model'
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main=nn.Sequential(
            
            nn.Conv2d(3,64,4,2,1,groups = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.Conv2d(64,128,4,2,1,groups = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.Conv2d(128,256,4,2,1,groups = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            
            nn.Conv2d(256,512,4,2,1,groups = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            
            nn.Conv2d(512,3,4,1,0,groups = 1),  #1*1
            nn.Sigmoid()
            
        )
        
        kenel_size = [4,4,4,4,4]
        stride = [2,2,2,2,1]
        padding = [1,1,1,1,0]
        output_height = IMG_HEIGHT
        output_width = IMG_WIDTH
        
        for i in range(5):
            output_height = (output_height-kenel_size[i]+2*padding[i])/stride[i]+1
            output_width = (output_width-kenel_size[i]+2*padding[i])/stride[i]+1
            
        self.fc1 = nn.Sequential(
            nn.Linear(3*int(output_height)*int(output_width),1),
            nn.Sigmoid()
        )
       
        self.fc2 = nn.Sequential(
            nn.Linear(200,10),
            nn.Sigmoid(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(10,2),
            nn.Sigmoid(),
        )
              
    def forward(self, x):
        x = self.main(x)
        x = x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        print(x.shape)
        return x

netD = Discriminator().to(device)
netD.apply(weights_init)

lr = 0.0002
# Initialize BCELoss function
criterion = nn.BCELoss()
msecriterion = nn.MSELoss()
l1criterion = nn.L1Loss()

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

# Training Loop
G_losses = []
D_losses = []
iters = 0
num_epochs = 100

print("Starting Training Loop...")
#Q = ['1.044','2.085','2.871','3.488','4.086','4.583','5.277','6.384','8.412','13.491']
Q = ['1.140', '2.004', '2.908', '3.399','4.061', '4.523','5.248', '6.298', '8.387', '13.352']
for q in Q:
    dataset = getdataset(q)
    dataloader = DataLoader(dataset,batch_size=2,shuffle=True)
    for epoch in range(num_epochs):
        for i, (feature,label) in enumerate(dataloader, 0):
            netG.train()
            netD.train()
            netD.zero_grad()
            optimizerD.zero_grad()
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            real_images = label.to(device)
            compressed_img = feature.to(device)
            D_out_real = netD(real_images)
            print(D_out_real.shape)
            
            valid = torch.FloatTensor(np.random.uniform(low=0.9, high=1, size=(real_images.size(0)))).to(device)
            Loss_D_real = criterion(D_out_real,valid)

            fake_images = netG(compressed_img)
            fake = torch.FloatTensor(np.random.uniform(low=0, high=0.1, size=(real_images.size(0)))).to(device)
            D_out_fake = netD(fake_images.detach())
            Loss_D_fake = criterion(D_out_fake, fake)
            
            Loss_D = Loss_D_real + Loss_D_fake
            Loss_D.backward()
            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            optimizerG.zero_grad()
            G_out = netD(fake_images)
            Loss_G = criterion(G_out, valid)
            Loss_G.backward()
            optimizerG.step()
            
#             fake_image = fake_images[0].detach()
#             fake_image = fake_image.cpu()
#             plt.imshow(np.transpose(fake_image,(1,2,0)))
            
            # Output training stats
            if i % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                      % (epoch, num_epochs, i, len(dataloader),
                         Loss_D, Loss_G))
                
                fake_image = fake_images[0].cpu().detach().numpy()
                min_val = np.min(fake_image)
                max_val = np.max(fake_image)
                fake_image = (fake_image - min_val) / (max_val - min_val)
                plt.imsave(str(i)+str(epoch)+'.png',np.transpose(fake_image,(1,2,0)))
                
            G_losses.append(Loss_G)
            D_losses.append(Loss_D)
   
            del real_images
            del fake_images
            del feature
            del label
            del G_out
            del D_out_fake
            
            torch.cuda.empty_cache()
            iters += 1

torch.save(netG.state_dict(), "netG"+str(3)+".model")
torch.save(netD.state_dict(), "netD"+str(3)+".model")