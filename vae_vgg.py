import torchvision
import PIL, image
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms, datasets, models
from torchvision.utils import save_image
from custom_dataset import CustomDatasetFromImages
import numpy as np
import matplotlib.pyplot as plt


def train(epoch):
    # toggle model to train mode
    model.train()
    train_loss = 0
    # in the case of MNIST, len(train_loader.dataset) is 60000
    # each `data` is of BATCH_SIZE samples and has shape [128, 1, 28, 28]
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        # push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, logvar = model(data)
        # calculate scalar loss
        loss = loss_function(recon_batch, data, mu, logvar)
        # calculate the gradient of the loss w.r.t. the graph leaves
        # i.e. input variables -- by the power of pytorch!
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    
    if epoch % 5 == 0:
        torch.save(model.state_dict(), './model_{}.pth'.format(epoch))


def test(epoch):
    # toggle model to test / inference mode
    model.eval()
    test_loss = 0

    # each data is of BATCH_SIZE (default 128) samples
    for i, (data, _) in enumerate(test_loader):
        with torch.no_grad():
          data = data.to(device)
          recon_batch, mu, logvar = model(data)
          test_loss += loss_function(recon_batch, data, mu, logvar).item()
          if i == 0:
            n = min(data.size(0), 24)
            # for the first 128 batch of the epoch, show the first 8 input digits
            # with right below them the reconstructed output digits
            data[:n].size
            comparison = torch.cat([data[:n],
                                    recon_batch.view(BATCH_SIZE, 3, 250, 250)[:n]])
            save_image(comparison.data.cpu(),
                       'results/reconstruction_' + str(epoch) + '.png', nrow=n)
        if i > int(len(test_loader)/2):
           break

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))



class Interpolate(nn.Module):
  def forward(self, x):
    return F.interpolate(x, scale_factor=2)


def make_decoder_layers(batch_norm=True):
    cfg = reversed([3 ,64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 'M'])
    layers = []
    in_channels = 512
    
    for v in cfg:
        if v == 'M':
            layers += [Interpolate()]
        else:
            unconv2d = nn.ConvTranspose2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [unconv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [unconv2d, nn.ReLU(inplace=True)]
            in_channels = v
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)
        
class VAE(nn.Module):
    def __init__(self, vgg):
        super(VAE, self).__init__()
        
        self.vgg = vgg    
        self.fc_mu = nn.Linear(512 * 7 * 7, 4096)
        self.fc_var = nn.Linear(512 * 7 * 7, 4096)
        self.fc_dec = nn.Linear(4096, 512 * 7 * 7)
        self.decode = make_decoder_layers()
        
    def encode(self, x):
        x = self.vgg.features(x)
        x = x.view(-1, np.prod(x.shape[1:]))
        mu, logvar = self.fc_mu(x), self.fc_var(x)
        return mu, logvar
      
      
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = self.fc_dec(z)
        z = z.view(-1,512,7,7)
        return self.decode(z), mu, logvar



def loss_function(recon_x, x, mu, logvar) -> Variable:
    BCE = F.binary_cross_entropy(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= BATCH_SIZE * (32*32*3)

    return BCE.mul(0.9) + KLD.mul(0.1)


CUDA = True
SEED = 1
BATCH_SIZE = 12
LOG_INTERVAL = 10
EPOCHS = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_loader = torch.utils.data.DataLoader(
            CustomDatasetFromImages('lfw', downsample=True),
                batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    CustomDatasetFromImages('lfw', downsample=True),
    batch_size=BATCH_SIZE, shuffle=True)


vgg = torchvision.models.vgg11_bn(pretrained=True)
model = VAE(vgg).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for i in range(1, EPOCHS + 1):
    train(i)
    test(i)




