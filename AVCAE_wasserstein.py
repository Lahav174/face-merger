# example from https://github.com/pytorch/examples/blob/master/vae/main.py
# commented and type annotated by Charl Botha <cpbotha@vxlabs.com>
import os
import torch
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
from custom_dataset import CustomDatasetFromImages

#VAE and general params
CUDA = True
SEED = 1
BATCH_SIZE = 64
LOG_INTERVAL = 10
EPOCHS = 100

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

# Download or load downloaded MNIST dataset
# shuffle data at every epoch


train_loader = torch.utils.data.DataLoader(
    CustomDatasetFromImages('lfw'),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

# Same for test data
test_loader = torch.utils.data.DataLoader(
    CustomDatasetFromImages('lfw'),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)


clamp_lower = 0.01
clamp_upper = 0.01

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 7, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 7)
        self.conv3 = nn.Conv2d(8, 16, 9)
        self.conv4 = nn.Conv2d(16, 16, 9)
        self.conv5 = nn.Conv2d(16, 32, 9)
        self.conv6 = nn.Conv2d(32, 32, 9)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.BatchNorm2d(8)
        self.norm2 = nn.BatchNorm2d(16)
        self.norm3 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(32 * 18 * 18, 250)
        self.fc2 = nn.Linear(250, 80)
        self.fc3 = nn.Linear(80, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 18 * 18)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
                
        return x

class VAE(nn.Module):
    
    def __init__(self):
        super(VAE, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 8, 7, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 7)
        self.conv3 = nn.Conv2d(8, 16, 9)
        self.conv4 = nn.Conv2d(16, 16, 9)
        self.conv5 = nn.Conv2d(16, 32, 9)
        self.conv6 = nn.Conv2d(32, 32, 9)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.BatchNorm2d(8)
        self.norm2 = nn.BatchNorm2d(16)
        self.norm3 = nn.BatchNorm2d(32)
        
        self.fc11 = nn.Linear(32 * 18 * 18, 5000)
        self.fc12 = nn.Linear(32 * 18 * 18, 5000)
        self.fc2 = nn.Linear(5000, 32 * 18 * 18)
        
        self.unconv6 = nn.ConvTranspose2d(32, 32, 9)
        self.unconv5 = nn.ConvTranspose2d(32, 16, 9)
        self.unconv4 = nn.ConvTranspose2d(16, 16, 9)
        self.unconv3 = nn.ConvTranspose2d(16, 8, 9)
        self.unconv2 = nn.ConvTranspose2d(8, 8, 7)
        self.unconv1 = nn.ConvTranspose2d(8, 3, 7, padding=1)
        self.interpolate = F.interpolate
        self.sigmoid = nn.Sigmoid()
        self.norm4 = nn.BatchNorm2d(16)
        self.norm5 = nn.BatchNorm2d(8)
        self.norm6 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()
                
    def encode(self, x: Variable) -> (Variable, Variable):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pool(x)
        assert x.shape[2]%2 == 0, "The width and height of the image are not even"
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pool(x)
        assert x.shape[2]%2 == 0, "The width and height of the image are not even"
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 18 * 18)
        mu, logvar = self.fc11(x), self.fc12(x)
        return mu, logvar
    
    def decode(self, x: Variable) -> Variable:
        x = self.fc2(x)
        x = x.view(-1, 32, 18, 18)
        x = self.interpolate(x,scale_factor=2)
        x = self.unconv6(x)
        x = self.unconv5(x)
        x = self.norm4(x)
        x = self.relu(x)
        x = self.interpolate(x,scale_factor=2)
        x = self.unconv4(x)
        x = self.unconv3(x)
        x = self.norm5(x)
        x = self.relu(x)
        x = self.interpolate(x,scale_factor=2)
        x = self.unconv2(x)
        x = self.unconv1(x)
        x = self.norm6(x)       
        x = self.sigmoid(x)
        assert x.shape[1:4] == torch.Size([3, 250, 250])
        return x
    

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        """THE REPARAMETERIZATION IDEA:
        For each training sample (we get 128 batched at a time)
        - take the current learned mu, stddev for each of the ZDIMS
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see generator_loss_function() below)
          the distribution will tend to unit Gaussians
        Parameters
        ----------
        mu : [128, ZDIMS] mean matrix
        logvar : [128, ZDIMS] variance matrix
        Returns
        -------
        During training random sample from the learned ZDIMS-dimensional
        normal distribution; during inference its mean.
        """

        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation
            std = logvar.mul(0.5).exp_()  # type: Variable
            # - std.data is the [128,ZDIMS] tensor that is wrapped by std
            # - so eps is [128,ZDIMS] with all elements drawn from a mean 0
            #   and stddev 1 normal distribution that is 128 samples
            #   of random ZDIMS-float vectors
            eps = Variable(std.data.new(std.size()).normal_())
            # - sample from a normal distribution with standard
            #   deviation = std and mean = mu by multiplying mean 0
            #   stddev 1 sample with desired std and mu, see
            #   https://stats.stackexchange.com/a/12338
            # - so we have 128 sets (the batch) of random ZDIMS-float
            #   vectors sampled from normal distribution with learned
            #   std and mu for the current input
            return eps.mul(std).add_(mu)

        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


generator = VAE()
discriminator = Discriminator()
if CUDA:
    generator.cuda()
    discriminator.cuda()


def generator_loss_function(recon_x, x, mu, logvar) -> Variable:
    # how well do input x and output recon_x agree?
    MSE = F.mse_loss(recon_x, x)

    # KLD is Kullback–Leibler divergence -- how much does one learned
    # distribution deviate from another, in this specific case the
    # learned distribution from the unit Gaussian

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # note the negative D_{KL} in appendix B of the paper
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= BATCH_SIZE * (250*250*3)

    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    return MSE.mul(0.5) + KLD.mul(0.5)

adversarial_loss = torch.nn.BCELoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr = 0.00005)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = 0.0005)

Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor





train_gen_losses = []
train_dis_losses = []
def train(epoch):    
    generator.train()
    discriminator.train()
    running_gen_loss = 0.0
    running_dis_loss = 0.0
    
    
    data_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):   
        
        if i%30==0:
            print("Batch number",i)
        
        ############################
        # (1) Update D network
        ###########################
        for p in discriminator.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in generator update

        # train the discriminator Diters times
        Diters = 100
        j = 0
        while j < Diters and i < len(train_loader):
            j += 1

            # clamp parameters to a cube
            for p in discriminator.parameters():
                p.data.clamp_(clamp_lower, clamp_upper)

            i += 1

            data = data_iter.next()
            inputs, _ = data
            batch_size = inputs.size(0)
            one = torch.FloatTensor(np.ones(batch_size).reshape((batch_size,1)))
            mone = one * -1
            if CUDA:
                one, mone = one.cuda(), mone.cuda()
                inputs = inputs.cuda()
            recon_img, _, _ = generator(inputs)
            
            discriminator.zero_grad()
                
            #input.resize_as_(inputs).copy_(inputs)
            inputv = Variable(inputs.clone())

            errD_real = discriminator(inputv)
            errD_real.backward(one)

            # train with fake
            inputv = recon_img.clone()
            errD_fake = discriminator(inputv)
            errD_fake.backward(mone)
            running_dis_loss += (errD_real - errD_fake)#FOR PRINTING
            optimizer_D.step()

            #print("Finished discrim")    
        ############################
        # (2) Update G network
        ###########################
        for p in discriminator.parameters():
            p.requires_grad = False # to avoid computation
        optimizer_G.zero_grad()

        data = data_iter.next()
        inputs, _ = data
        batch_size = inputs.size(0)
        one = torch.FloatTensor(np.ones(batch_size).reshape((batch_size,1)))
        mone = one * -1
        if CUDA:
                one, mone = one.cuda(), mone.cuda()
                inputs = inputs.cuda()
        
        recon_img, mu, logvar = generator(inputs)
        g_loss = generator_loss_function(recon_img, inputs, mu, logvar)
        running_gen_loss += g_loss.item()
        g_loss.backward()
        
        discrim_input = Variable(recon_img.detach(), requires_grad=True)
        g_loss = discriminator(discrim_input)
        running_gen_loss += g_loss.mean().item()
        g_loss.backward(one)
        
        optimizer_G.step()   
        #print("Finished generator")  
    
    
    train_dis_losses.append(running_dis_loss / len(train_loader))
    train_gen_losses.append(running_gen_loss / len(train_loader))
    
    if not os.path.isdir("results"):
        os.mkdir("results")
    
    np.savetxt('results/train_gen_losses.txt', np.array(train_gen_losses))
    np.savetxt('results/train_dis_losses.txt', np.array(train_dis_losses))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
def test(epoch):
    if not os.path.isdir("results"):
        os.mkdir("results")
    
    generator.eval()
    discriminator.eval()
    #test_loss = 0

    for i, (inputs, _) in enumerate(test_loader):

        inputs = Variable(inputs)
        
        if CUDA:
            inputs = inputs.cuda()
            
        
        recon_img, mu, logvar = generator(inputs)
            
        #test_loss += generator_loss_function(recon_img, inputs, mu, logvar).item()
        if i == 0:
          n = min(inputs.size(0), 8)
          # for the first 128 batch of the epoch, show the first 8 input digits
          # with right below them the reconstructed output digits
          comparison = torch.cat([inputs[:n],
                                  recon_img.view(BATCH_SIZE, 3, 250, 250)[:n]])
          save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    #test_loss /= len(test_loader.dataset)
    #print('====> Test set loss: {:.4f}'.format(test_loss))



for epoch in range(1, EPOCHS + 1):
    print("Epoch:",epoch)
    train(epoch)
    test(epoch)

    