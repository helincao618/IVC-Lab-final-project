import numpy as np 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import ImageData
from model import Encoder, Generator, Discriminator

device = 'cpu'
img_dir = 'dataset/'
imagePaths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
img_list = os.listdir(img_dir)
valid_ratio = 0.8
batch_size=2
IMG_WIDTH = 288
IMG_HEIGHT = 352
num_channels_in_encoder = 3
lr = 0.0002
num_epochs = 100

def getdataset(Q):
    featpath2 = './dataset/compression_ratio_'
    labelpath2 = './dataset/Groundtruth/'
    train = ImageData(featpath2+Q,labelpath2)
    return train

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:               #initial cov layer
        nn.init.normal_(m.weight.data, 0.0, 0.02)  
    elif classname.find('BatchNorm') != -1:        #initial BN layer
        nn.init.normal_(m.weight.data, 1.0, 0.02)  
        nn.init.constant_(m.bias.data, 0)          #setting bias to constant

netE = Encoder().to(device)
netE.apply(weights_init)
input = torch.randn(IMG_WIDTH*IMG_HEIGHT*3 * 1)
input = input.view((-1,3,IMG_HEIGHT,IMG_WIDTH))
input = input.to(device)
output = netE(input)

netG = Generator().to(device)
netG.apply(weights_init)
input = torch.randn(2*num_channels_in_encoder*IMG_WIDTH*IMG_HEIGHT)
input = input.view((-1,3,IMG_HEIGHT,IMG_WIDTH))
input = input.to(device)
output = netG(input)
torch.cuda.empty_cache()
output = output.clamp(0.0, 1.0)
output = output.cpu() 

netD = Discriminator(IMG_WIDTH,IMG_HEIGHT).to(device)
netD.apply(weights_init)

# Initialize BCELoss function
criterion = nn.BCELoss()
msecriterion = nn.MSELoss()
l1criterion = nn.L1Loss()

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

# Training Loop
G_losses = []
D_losses = []
iters = 0
print("Starting Training Loop...")
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
            if i % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                      % (epoch, num_epochs, i, len(dataloader),
                         Loss_D, Loss_G))
                
                fake_image = fake_images[0].cpu().detach().numpy()
                min_val = np.min(fake_image)
                max_val = np.max(fake_image)
                fake_image = (fake_image - min_val) / (max_val - min_val)
                # plt.imsave(str(i)+str(epoch)+'.png',np.transpose(fake_image,(1,2,0)))
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
