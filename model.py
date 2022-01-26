import torch.nn as nn

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
        return dc4

'Discriminator Model'
class Discriminator(nn.Module):
    def __init__(self,output_width,output_height):
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
        return x