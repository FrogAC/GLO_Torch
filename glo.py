from __future__ import print_function
import os
import argparse as arg

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from torch.autograd import Variable

# gpu
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Parameters
batch_size = 128
latent_size = 2
image_size = 32
image_channels = 1

learning_rate_net = 2e-4 #5e-4
learning_rate_latent = 1e-3 #1e-3
adam_momentum = 0.5 # 0.5

class Generator(nn.Module):
    def __init__(self):
       super(Generator, self).__init__()
       self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_size, image_size * 4, 4, 1, 0, bias = True),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(True),
            # 4
            nn.ConvTranspose2d(image_size * 4, image_size *2, 4, 2, 1, bias = True),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(True),
            # 8
            nn.ConvTranspose2d(image_size * 2, image_size, 4, 2, 1, bias = True),
            nn.BatchNorm2d(image_size),
            nn.ReLU(True),
            # 16
            nn.ConvTranspose2d(image_size, image_channels, 4, 2, 1, bias = True),
            nn.Tanh(),
            # img_channel * 32 * 32
       )
    
    def forward(self, input):
        return self.net(input)

class IndexedMNIST(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.mnist = dset.MNIST(**kwargs)
        
    def __getitem__(self, index):
        data, label = self.mnist[index]
        return data, label, index

    def __len__(self):
        return len(self.mnist)

def tanhScale(x):
    return x * 2 - 1

# custom data loader
def get_loader(train = True, batch = True):
    # load data, mnist from -1 to 1 to match tanh
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(), # [0,1]
        transforms.Lambda(tanhScale) # [-1,1]
        ])
    dataset = IndexedMNIST(root = './data/mnist', download = True, transform = image_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size if batch else 1, shuffle = True, num_workers = 4, drop_last = True)
    return dataset, dataloader

# inline projection. z: (b_size, d, 1, 1)
def projectl2(z):
    norm = torch.sqrt(torch.sum(z**2, axis=1))[:,np.newaxis]
    return z/torch.max(torch.ones_like(norm), norm)

# for conv and batch-normlize
def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.BatchNorm2d]:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

def get_file_path(name):
    path = os.path.join('.','save')
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name+'.pth')
    return path

def test(num_epochs, name):
    saved = torch.load(get_file_path(name))

    # dataset
    dataset, dataloader = get_loader(train = False, batch = False)

    # networks
    netG =Generator().to(device)
    netG.load_state_dict(saved['model'])
    netG.eval()

    criterion = nn.MSELoss().to(device)

    loss_list = []
    img_list = []

    # force batch 1
    batch_size = 1

    for i, (batch_data, _, _) in enumerate(dataloader):
        if i > 7: break
        # init zero latent
        latent_z = torch.randn(batch_size, latent_size, 1, 1, device = device)
        latent_z = projectl2(latent_z)

        learnable_z = torch.zeros((batch_size,latent_size), device=device, requires_grad=True)
        
        # optimizer
        optimizer_test = optim.Adam([ {'params':learnable_z, 'lr': 0.01, 'betas':(adam_momentum, .999)} ])
        
        real_images = batch_data.to(device).view(-1)

        for epoch in range(num_epochs):
            # prepare grad
            learnable_z.data = latent_z
            optimizer_test.zero_grad()
            gen_images = netG(learnable_z)

            # back prob for z
            loss = criterion(real_images, gen_images.view(-1))
            loss.backward()
            optimizer_test.step()

            # project z and update
            latent_z = projectl2(learnable_z)

            # save loss
            loss_list.append(loss.item())

        # output stats
        # if epoch % 20 == 0:
        print('[{:d}]\tloss: {:.4f}'.format(i, loss))

        # save latent_z for visualization
        recon_img = gen_images.detach().view((image_channels,image_size,image_size)).cpu()
        target_image = real_images.view((image_channels,image_size,image_size)).cpu()
        img_list.append(vutils.make_grid([recon_img, target_image], padding = 1, normalize = True))
    return img_list, loss_list

def train(num_epochs, name, hot_start = False):
    # dataset
    dataset, dataloader = get_loader(train = True, batch = True)

    netG = Generator().to(device)

    # graded z used for learning
    batch_z = torch.zeros((batch_size,latent_size), device=device, requires_grad=True)

    # l2 loss
    criterion = nn.MSELoss().to(device)

    # optimizer
    optimizer = optim.Adam([
        {'params':netG.parameters(), 'lr': learning_rate_net, 'betas':(adam_momentum, .999)},
        {'params':batch_z, 'lr': learning_rate_latent, 'betas':(adam_momentum, .999)}
    ])

    # list for stats
    img_list = []
    loss_list = []

    # init net, latent, optimizer
    if not hot_start:
        netG.apply(weights_init)
        latent_z = torch.randn(len(dataset), latent_size, 1, 1, device = device)
        latent_z = projectl2(latent_z)
        epoch_start = 0
    else:
        saved = torch.load(get_file_path(name))
        netG.load_state_dict(saved['model'])
        netG.train()
        latent_z = saved['latent']
        latent_z.requires_grad = False
        optimizer.load_state_dict(saved['optimizer'])
        epoch_start = saved['epoch']

    # with torch.autograd.set_detect_anomaly(True):
    for epoch in range(epoch_start, num_epochs):
        avgloss = 0
        for i, (batch_data,_, idx) in enumerate(dataloader):
            # extract batch and prepare z for optim
            batch_z.data = latent_z[idx]
            optimizer.zero_grad()

            gen_images = netG(batch_z)
            real_images = batch_data.to(device)
            loss = criterion(real_images.view(-1), gen_images.view(-1))

            # step
            loss.backward()
            optimizer.step()

            # project z after update
            latent_z[idx] = projectl2(batch_z)

            # save loss
            avgloss += loss.item()

        # sates save
        avgloss /= len(dataloader)
        loss_list.append(avgloss)
        print('[{:d}]\tloss: {:.4f}'.format(epoch, avgloss))

        if epoch % 10 == 0:
            # save check points
            torch.save({
                'epoch': epoch + 1,
                'model': netG.state_dict(),
                'optimizer': optimizer.state_dict(),
                'latent': latent_z
            }, get_file_path(name))

            # store image result
            with torch.no_grad():
                outputG = netG(latent_z[[999,888,777,666]].reshape(4,latent_size,1,1).detach()).cpu()
            img_list.append(vutils.make_grid(outputG, padding = 2, normalize = True))

    # save result to file
    torch.save({
        'epoch': num_epochs,
        'model': netG.state_dict(),
        'optimizer': optimizer.state_dict(),
        'latent': latent_z
    }, get_file_path(name))

    return img_list, loss_list

def vis(name, row_col):
    saved = torch.load(get_file_path(name))
    
    netG =Generator().to(device)
    netG.load_state_dict(saved['model'])
    netG.eval()
    
    # only suppor 2d latent
    assert saved['latent'].shape[1] == 2

    # vis the latent disk on grid
    fig = plt.figure(figsize=(12,12))

    row = row_col
    col = row_col
    for i in range(row):
        for j in range(col):
            ii = (i / row) * 2 - 1
            jj = (j / col) * 2 - 1
            # if ii ** 2 + jj ** 2 > 10 :
            #     npimg = np.zeros((image_size, image_size))
            # else:
            latent = torch.FloatTensor([ii,jj]).to(device)
            cpuimg = netG(latent.view((1,2,1,1))).view((image_size, image_size)).detach().cpu()
            npimg = cpuimg.numpy()
            npimg = (npimg + 1) / 2  # [0,1]

            ax = fig.add_subplot(row, col, i*col+j+1)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(npimg)

    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.tight_layout(pad=0.0)
    plt.show()

if __name__ == '__main__':
    parser = arg.ArgumentParser()
    parser.add_argument('--vis',action='store_true', help='visulize 2d latent space')
    parser.add_argument('--test',action='store_true', help='run test')
    parser.add_argument('--load',action='store_true', help='hot start')
    parser.add_argument('name', type=str)
    parser.add_argument('epoch',type=int, help = 'r')
    args = parser.parse_args()

    if args.vis:
        vis(args.name, args.epoch)
    else:
        if not args.test:
            res_imgs, res_loss = train(args.epoch, args.name, hot_start = args.load)
            # animation G
            fig = plt.figure(figsize=(8,8))
            plt.axis("off")
            ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in res_imgs]
            ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
            plt.show()
        else:
            res_imgs, res_loss = test(args.epoch, args.name)
            # visulize in pairs
            fig=plt.figure(figsize=(8, 8))
            rows = 8
            for i in range(rows):
                img = np.transpose(res_imgs[i], (1,2,0))
                print(img.shape)
                fig.add_subplot(rows, 1, i + 1)
                plt.imshow(img)
            fig.tight_layout(pad=0.0)
            plt.show()

        # plot loss
        plt.figure(figsize=(10,5))
        plt.plot(res_loss,label="loss")
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.legend()
        plt.show()
