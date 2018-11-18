import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

#I based the GAN setup off of the code here: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/acgan/acgan.py

CUDA = False
BATCH_SIZE = 128
EPOCHS = 100
SEED = 1

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}


# Download or load downloaded CIFAR dataset
# shuffle data at every epoch
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

# Same for test data
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=False, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    
    

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 6, 5, padding=2)
        self.conv3 = nn.Conv2d(6, 6, 5, padding=2)
        self.conv4 = nn.Conv2d(6, 16, 5, padding=2)
        self.norm6 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2)
        self.norm16 = nn.BatchNorm2d(16)
        
        
        self.unconv4 = nn.ConvTranspose2d(16, 6, 5, padding=2)
        self.unconv3 = nn.ConvTranspose2d(6, 6, 5, padding=2)
        self.unconv2 = nn.ConvTranspose2d(6, 6, 5, padding=2)
        self.unconv1 = nn.ConvTranspose2d(6, 3, 5, padding=2)
        self.interpolate = F.interpolate
        self.sigmoid = nn.Sigmoid()
        self.bn6 = nn.BatchNorm2d(6)
        self.bn3 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()
        
        initialize_weights(self)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm6(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.norm16(x)
        x = self.relu(x)
        x = self.pool(x)
        
        
        x = self.interpolate(x,scale_factor=2)
        x = self.unconv4(x)
        x = self.unconv3(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.interpolate(x,scale_factor=2)
        x = self.unconv2(x)
        x = self.unconv1(x)
        x = self.bn3(x)       
        decoded = self.sigmoid(x)
        
        return decoded

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        
generator = AE()
discriminator = Discriminator()
if CUDA:
    generator.cuda()
    discriminator.cuda()
        

# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr = 0.002)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = 0.002)

Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

def train(epoch):    
    generator.train()
    discriminator.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):    
        
        # Adversarial ground truths
        valid = Variable(Tensor(inputs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(inputs.shape[0], 1).fill_(0.0), requires_grad=False)
        
        # get the inputs
        inputs = Variable(inputs)
        labels = inputs.clone()
        
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        # -----------------
        #  Train Generator
        # -----------------
        
            
        
        optimizer_G.zero_grad()

        recon_img = generator(inputs)

#        loss = criterion(outputs, labels)
        # Loss measures generator's ability to fool the discriminator
        g_loss =    0.001 * adversarial_loss(discriminator(recon_img), valid) + \
                    0.999 * pixelwise_loss(recon_img, labels)
        
        g_loss.backward()
        optimizer_G.step()
        
        running_loss += g_loss.item()
        
        ave_loss = running_loss / (i + 1.0)
        if i%80==0:
            print('average loss is: ', str(ave_loss))
            
            
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(inputs), valid)
        fake_loss = adversarial_loss(discriminator(recon_img.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()        
    
            
def test(epoch):
    if not os.path.isdir("results"):
        os.mkdir("results")
    
    generator.eval()
    discriminator.eval()
    test_loss = 0

    for i, (inputs, labels) in enumerate(test_loader):

        inputs = Variable(inputs)
        labels = inputs.clone()
        
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()
            
        recon_img = generator(inputs)  
            
        test_loss += pixelwise_loss(recon_img, labels)
        if i == 0:
          n = min(inputs.size(0), 8)
          # for the first 128 batch of the epoch, show the first 8 input digits
          # with right below them the reconstructed output digits
          comparison = torch.cat([inputs[:n],
                                  recon_img.view(BATCH_SIZE, 3, 32, 32)[:n]])
          save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))



for epoch in range(1, EPOCHS + 1):
    print("Epoch:",epoch)
    train(epoch)
    test(epoch)



    