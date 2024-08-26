# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 20:33:55 2024

@author: Priyanshu singh
"""
import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
   
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

trainset = datasets.MNIST(root="./data",train=True,transform=transform,download=True)

trainloader = DataLoader(dataset=trainset,batch_size=60,shuffle=True,num_workers=0)

testset = datasets.MNIST(root = "./data",train = False,transform=transform,download=True)

testloader = DataLoader(dataset=testset,batch_size =50,shuffle = True,num_workers = 0)


#%%
class Embedding(nn.Module):
    def __init__(self,num_classes,embedding_out):
        super(Embedding,self).__init__()
        self.num_classes = num_classes
        self.embedding_out = embedding_out
        self.main = nn.Sequential(nn.Embedding(self.num_classes, self.embedding_out),
                                  )
        
    def forward(self,index):
        return self.main(index).unsqueeze(2).unsqueeze(3)
embed = Embedding(num_classes=10, embedding_out=20)
tensor_1 = torch.tensor([0,1,3])
new1 = embed(tensor_1)
print(new1.size())
#%%

latent_dim = 120
ngf = 28
torch.cuda.empty_cache()

num_classes = trainloader.dataset.classes
class Generator(nn.Module):
    
    def __init__(self,latent_dim,ngf):
        self.latent_dim = latent_dim
        self.ngf = ngf
        super(Generator,self).__init__()
        self.main = nn.Sequential(
                    
                    nn.ConvTranspose2d(self.latent_dim, self.ngf*32, kernel_size=4,padding=0,stride = 1,bias=False),
                    
                    nn.ReLU(True),
                    
                    
                    nn.utils.spectral_norm(nn.ConvTranspose2d(self.ngf*32, self.ngf*16, kernel_size=4,padding=0,stride = 1,bias=False)),
                    
                    nn.ReLU(True),
                    
                    nn.utils.spectral_norm(nn.ConvTranspose2d(self.ngf*16, self.ngf*8, kernel_size=4,padding=1,stride = 2,bias=False)),
                    
                    nn.ReLU(True),
                   
                    nn.utils.spectral_norm(nn.ConvTranspose2d(self.ngf*8, self.ngf*4, kernel_size=4,padding=1,stride = 2,bias=False)),
                    
                    nn.ReLU(True),
                    
                    nn.ConvTranspose2d(self.ngf*4, 1, kernel_size=3,padding=1,stride = 1,bias=False),
                    nn.Tanh(),
                    
            )
    def forward(self,input):
        return self.main(input)
    
rand_tensor = torch.randn(60,latent_dim,1,1).to(device)

gen = Generator(latent_dim,ngf).to(device)
print(gen)

output =gen(rand_tensor)
print(output.size())

#output = output.squeeze(0).permute(1,2,0).detach().numpy().astype(np.float32)

#print(np.max(output))

#plt.imshow(output)
#%%
col_channel=21
ndf = 28
torch.cuda.empty_cache()

class Discriminator(nn.Module):
    def __init__(self,col_channel,ndf):
        self.col_channel = col_channel
        self.ndf = ndf
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(
                    nn.utils.spectral_norm(nn.Conv2d(in_channels=self.col_channel,out_channels=self.ndf*2,kernel_size=4,padding=1,stride = 2,bias= False)),
                    
                    nn.LeakyReLU(0.2,inplace=True),
                    
                    nn.utils.spectral_norm(nn.Conv2d(in_channels=self.ndf*2,out_channels=self.ndf*4,kernel_size=4,padding=0,stride=2,bias= False)),
                    
                    nn.LeakyReLU(0.2,inplace=True),
                    nn.Dropout2d(0.175),
                    
                    nn.utils.spectral_norm(nn.Conv2d(in_channels=self.ndf*4,out_channels=self.ndf*8,kernel_size=4,padding=0,stride=2,bias= False)),
                    
                    nn.LeakyReLU(0.2,inplace=True),
                    nn.Dropout2d(0.175),
                    
                    nn.utils.spectral_norm(nn.Conv2d(in_channels=self.ndf*8,out_channels=self.ndf*16,kernel_size=3,padding=1,stride=2,bias= False)),
                    
                    nn.LeakyReLU(0.2,inplace=True),
                    nn.Dropout2d(0.315),
                    
                    nn.Flatten(),
                    nn.Linear(in_features=448, out_features=100,bias = False),
                    
                    nn.LeakyReLU(0.2,inplace=True),
                    nn.Dropout(0.15),

                    nn.Linear(in_features=100, out_features=10,bias = False),
                    
                    
                    
            )
    def forward(self,input):
        
        return self.main(input)
    
    
disc = Discriminator(col_channel, ndf).to(device)
print(disc)
rand_tensor = torch.randn(60,21,28,28).to(device)

output = disc(rand_tensor)
print(output.size())

#%%


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        
gen.apply(weights_init)
disc.apply(weights_init)
num_epoch = 50
ler_g = 0.00120
ler_d = 0.00030
g_loss = []
d_loss = []
img_list = []
embed_class = Embedding(num_classes=10, embedding_out=20).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer_d = optim.AdamW(disc.parameters(),lr=ler_d)
optimizer_g = optim.AdamW(gen.parameters(),lr=ler_g)
optimizer_e = optim.AdamW(embed_class.parameters(),lr = 0.0004)
scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=2,gamma=0.1)
scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=1,gamma=0.1)


torch.cuda.empty_cache()



for epoch in range(num_epoch):
    
    for i,(data,label) in enumerate(trainloader):
        # DATA SIZE  60 X 1 X 64 X 64
        
        #label 60 x 10 
        label = label.to(device)
        #print("label size",label.size())
        data = data.to(device)
        real_tensor = torch.full((60,10),0.1,device = device)
        
        #embed made 60 x 20 x 1 x 1 now repeat will add and make it to 60 x 50 x 64 x 64
        emb_tensor = embed_class(label).repeat(1,1,28,28).to(device)
        
        #CONCAT will do 60 x 1+20 x 64 x 64
        real_withlabel = torch.cat((data,emb_tensor),dim=1).to(device)
        
        # SIZE 60 x 21 X 64 X 64
        
        disc.zero_grad()
        
        output = disc(real_withlabel)
        
        real_d_loss = criterion(output,real_tensor)
        real_d_loss.backward()
        
        
        
        noise = torch.randn(60,100,1,1,device=device)
        # rand label size 60 x 10 
        indices = torch.randperm(label.size(0))
        rand_label_tensor = label[indices]
        
        # TO CHECK IF .CLONE WILL CREATE NOT RANDOM NUMBER
        #print(rand_label_tensor)
        #print(rand_label_tensor)
        #print(rand_label_tensor)
        #print(rand_label_tensor)
        rand_emb_tensor = embed_class(rand_label_tensor).repeat(1,1,1,1).to(device)
        # got 60 x 50 x 1 x 1
        
        fake_withlabel =torch.concat((noise,rand_emb_tensor),dim=1)
        #print(fake_withlabel.size())
        # got 60 x 120 x 1 x 1
        
        fake_tensor = torch.full((60,10),0.9,device=device)
        fake_img = gen(fake_withlabel)
        
        emb_tensor = embed_class(label).repeat(1,1,28,28).to(device)
        #print(emb_tensor.size())
        #Size = 60 x 20 x 64 x 64
        #print(fake_img.size())
        #size = 60 x 1 x 64 x 64
        fake_withlabel = torch.cat((fake_img,emb_tensor),dim=1).to(device)
        #print(fake_withlabel.size())
        
        
        output = disc(fake_withlabel.detach())
        loss = criterion(output,fake_tensor)
        loss.backward()
        optimizer_d.step()
        total_d_loss = loss+real_d_loss
        
        
        
        
        
        gen.zero_grad()
        embed_class.zero_grad()
        
        output = disc(fake_withlabel)
        lossg= criterion(output,real_tensor)
        
        lossg.backward()
        
        optimizer_g.step()
        optimizer_e.step()
        
        g_loss.append(lossg.item())
        d_loss.append(total_d_loss.item())
        
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
              % (epoch, num_epoch, i, len(trainloader),
                 total_d_loss.item(), lossg.item()))
        if i%100==0:
            
            img_list.append(fake_img[30].detach().cpu().numpy())
            #fake_img = fake_img
            dis_img = fake_img[30].permute(1,2,0).cpu().detach().numpy()
            plt.imshow(dis_img,cmap="gray")
            plt.title(label[30].item())
            plt.show()
            
        #PROBABILITY DISTRIBUTION PLOTTING
        if i % 50 ==0:
            with torch.no_grad():
                fake_img = fake_img.detach().cpu().numpy().flatten()
                data = data.detach().cpu().numpy().flatten()
            plt.figure(figsize=(12, 6))
            sns.kdeplot(data, label='Real Data', fill=True, color='blue')
            sns.kdeplot(fake_img, label='Generated Data', fill=True, color='red')
        
            plt.title('Probability Distribution of Real and Generated Data')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.show()
                
                
    scheduler_g.step()
    scheduler_d.step()
            



#%%
y = list(range(len(g_loss)))

plt.plot(g_loss,label="Generator loss")
plt.plot(d_loss,label = "Discriminator los")
plt.legend()
plt.show()
 


