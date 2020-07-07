from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import matplotlib.animation as animation


# Hyper Parameters
batch_size = 64
learning_rate = 0.0002
adam_momentum = 0.5

# GAN Settings
latent_size = 100
feature_size_g = 64
feature_size_d = 64

class Generator(nn.Module):
    def __init__(self):
       super(Generator, self).__init__()
       self.net = nn.Sequential(
            # latent_size
            nn.ConvTranspose2d(latent_size, feature_size_g * 8, 4, 1,0, bias=False),
            nn.BatchNorm2d(feature_size_g * 8),
            nn.ReLU(True),
            # (feature_size_g * 8) * 4 * 4
            nn.ConvTranspose2d(feature_size_g * 8, feature_size_g *4, 4, 2, 1,bias = False),
            nn.BatchNorm2d(feature_size_g * 4),
            nn.ReLU(True),
            # (feature_size_g * 4) * 8 * 8
            nn.ConvTranspose2d(feature_size_g * 4, feature_size_g *2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(feature_size_g * 2),
            nn.ReLU(True),
            # (feature_size_g * 2) * 16 * 16
            nn.ConvTranspose2d(feature_size_g * 2, feature_size_g, 4, 2, 1, bias = False),
            nn.BatchNorm2d(feature_size_g),
            nn.ReLU(True),
            # (feature_size_g) * 32 * 32
            nn.ConvTranspose2d(feature_size_g, 3, 4, 2, 1, bias = False),
            nn.Tanh()
            # (feature_size_g) * 64 * 64
       )
    
    def forward(self, input):
        return self.net(input)

class Discriminator(nn.Module):
    def __init__(self):
       super(Discriminator, self).__init__()
       self.net = nn.Sequential(
            # 3 * 64 * 64
            nn.Conv2d(3, feature_size_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # feature_size_d * 32 * 32
            nn.Conv2d(feature_size_d, feature_size_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size_d * 2),
            nn.LeakyReLU(0.2, True),
            # (feature_size_d * 2) * 16 * 16
            nn.Conv2d(feature_size_d * 2, feature_size_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size_d * 4),
            nn.LeakyReLU(0.2, True),
            # (feature_size_d * 4) * 8 * 8
            nn.Conv2d(feature_size_d * 4, feature_size_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size_d * 8),
            nn.LeakyReLU(0.2, True),
            # (feature_size_d * 8) * 4 * 4
            nn.Conv2d(feature_size_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # feature_size_d * 8
       )
    
    def forward(self, input):
        return self.net(input)

# for conv and batch-normlize
def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train(netG, netD, num_epochs):
    # batch of latent vectors, used for visulize output
    fixed_latent_vector = torch.randn(batch_size, latent_size, 1, 1, device = device)

    optimG = optim.Adam(netG.parameters(), lr= learning_rate, betas=(adam_momentum, .999))
    optimD = optim.Adam(netD.parameters(), lr= learning_rate, betas=(adam_momentum, .999))

    # -w[y log(x) + (1-y) log(1-x)]
    criterion = nn.BCELoss().to(device)

    # list for stats
    img_list = []
    lossG_list = []
    lossD_list = []

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader):
                # extract batch
                real_images = data[0].to(device) # batch_size * 3 * 64 * 64
                b_size = real_images.shape[0]

                # 1 0 labels
                label_1 = torch.ones(b_size, device=device, requires_grad=False)
                label_0 = torch.zeros(b_size, device=device, requires_grad=False)

                ### G: max log(D(G(z)))
                ###   use label 1 for this loss

                netG.zero_grad()
                # noise z
                latent_vector = torch.randn(b_size, latent_size, 1, 1, device = device)
                # generate image batch
                fake_images = netG(latent_vector)
                # feed to netD, G(z) already calculated in fake_images
                outputs = netD(fake_images).reshape(-1)
                # loss with label = 1
                lossG = criterion(outputs, label_1)
                # step optim
                lossG.backward()
                optimG.step()

                ### D: max log(D(x)) + log(1 - D(G(z)))
                ###   use real sample x for loss[log(D(x))]
                ###   use generated sample z for loss[log(1-D(G(z)))]

                ## real batch
                netD.zero_grad()
                # feed batch to netD
                outputs = netD(real_images).reshape(-1)
                # real loss: log(D(x))
                loss_realD = criterion(outputs, label_1)

                ## fake batch
                # feed to D to classify. Detach since we'll need them one more time
                fake_detached = fake_images.detach()
                outputs = netD(fake_detached).reshape(-1)
                # fake loss: log(1-D(G(z)))
                loss_fakeD = criterion(outputs,label_0)

                # step optim
                lossD = loss_fakeD + loss_realD
                lossD.backward()
                optimD.step()
                
                # append to list
                lossG_list.append(lossG.item())
                lossD_list.append(lossD.item())
                
                # output stats
                if i % 50 == 0:
                    print('[{:d},{:d}]\td_loss: {:.4f}\tg_loss: {:.4f}'.format(epoch, i, lossD, lossG))

            # store result of each epoch
            with torch.no_grad():
                outputG = netG(fixed_latent_vector).detach().cpu()
            img_list.append(vutils.make_grid(outputG, padding = 2, normalize = True))

    return img_list, lossG_list, lossD_list
            


            
            

if __name__ == '__main__':
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # load data
    image_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5))])
    dataset = dset.CelebA(root = './data/celeba', download = True, transform = image_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 2)

    # networks
    netG =Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    # print(netG)
    # print(netD)

    # train!
    res_img, res_lossG, res_lossD = train(netG, netD, 30)

    # plot loss
    plt.figure(figsize=(10,5))
    plt.plot(res_lossG,label="G")
    plt.plot(res_lossD,label="D")
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    # animation G
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in res_img]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    plt.show()
