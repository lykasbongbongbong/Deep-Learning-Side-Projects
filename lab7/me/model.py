import torch 
import torch.nn as nn  

class Generator(nn.Module):
    def __init__(self, latent_dim, cond_dim, batchnorm=True):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim # 100
        self.cond_dim = cond_dim # 200
        self.batchnorm = batchnorm
        self._init_modules() 
    
    def _init_modules(self):
        # project inputs
        self.proj_reshape = nn.Sequential(
            nn.Linear(24, self.cond_dim),  # (24, c's dim)
            nn.ReLU()
        )
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim+self.cond_dim, 512, kernel_size=(4,4), stride=(2, 2), padding=(0,0)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(4,4), stride=(2, 2), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(4,4), stride=(2, 2), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(4,4), stride=(2, 2), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=(4,4), stride=(2,2), padding=(1,1))
        )
        self.tanh = nn.Tanh()
    
    def forward(self, z, c):
        z = z.view(-1, self.latent_dim, 1, 1)  
        c = self.proj_reshape(c).view(-1, self.cond_dim, 1, 1)    
        out = torch.cat((z, c), dim=1)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        out = self.tanh(out)
        return out 
    
    def weight_init(self,mean,std):
        for m in self._modules:
            if isinstance(self._modules[m], nn.ConvTranspose2d) or isinstance(self._modules[m], nn.Conv2d):
                self._modules[m].weight.data.normal_(mean, std)
                self._modules[m].bias.data.zero_()

class Discriminator(nn.Module):
    def __init__(self, img_shape, cond_dim):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape 
        self.num_classes = 24 
        self._init_modules() 

    def _init_modules(self):
        self.proj_reshape = nn.Sequential(
            nn.Linear(self.num_classes, self.img_shape[0]*self.img_shape[1]*self.img_shape[2]),
            nn.LeakyReLU()
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=(4,4), stride=(2,2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(4,4), stride=(2,2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(4,4), stride=(2,2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.deconv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(4,4), stride=(2,2), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.deconv5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=(4,4), stride=(1,1))
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, img, condition):
        c = self.proj_reshape(condition).view(-1, self.img_shape[2], self.img_shape[1], self.img_shape[0])
        out = torch.cat((img, condition), dim=1)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        out = self.sigmoid(out)
        out = out.view(-1)
        return out 
    
    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(self._modules[m], nn.ConvTranspose2d) or isinstance(self._modules[m], nn.Conv2d):
                self._modules[m].weight.data.normal_(mean, std)
                self._modules[m].bias.data.zero_()

        
        


