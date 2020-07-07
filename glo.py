from __future__ import print_function
import torch
import os
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import argparse as arg
from torch.autograd import Variable
import matplotlib.animation as animation
from torch.utils.data.dataloader import default_collate 

# gpu
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Parameters
batch_size = 64
latent_size = 100
feature_size = 64

learning_rate = 0.0002
adam_momentum = 0.5


class Generator(nn.Module):
    def __init__(self):
       super(Generator, self).__init__()
       out_channels = 1
       self.net = nn.Sequential(
            # latent_size
            nn.ConvTranspose2d(latent_size, feature_size * 8, 4, 1,0, bias=False),
            nn.BatchNorm2d(feature_size * 8),
            nn.ReLU(True),
            # (feature_size * 8) * 4 * 4
            nn.ConvTranspose2d(feature_size * 8, feature_size *4, 4, 2, 1,bias = False),
            nn.BatchNorm2d(feature_size * 4),
            nn.ReLU(True),
            # (feature_size * 4) * 8 * 8
            nn.ConvTranspose2d(feature_size * 4, feature_size *2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(feature_size * 2),
            nn.ReLU(True),
            # (feature_size * 2) * 16 * 16
            nn.ConvTranspose2d(feature_size * 2, feature_size, 4, 2, 1, bias = False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(True),
            # (feature_size) * 32 * 32
            nn.ConvTranspose2d(feature_size, out_channels, 4, 2, 1, bias = False),
            nn.Tanh()
            # (out_channels) * 64 * 64
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

# custom data loader
def get_loader(train = True, batch = True):
    # load data
    image_transform = transforms.Compose([
        transforms.Resize(feature_size),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5),std = (0.5))])
    dataset = IndexedMNIST(root = './data/mnist', download = True, transform = image_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size if batch else 1, shuffle = True, num_workers = 2, drop_last = True)
    return dataset, dataloader


# for conv and batch-normlize
def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_file_path(name):
    path = os.path.join('.','save')
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, 'name'+'.pth')
    # os.touch(path)
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
        latent_z /= torch.sqrt(torch.sum(latent_z**2, axis=1))[:,np.newaxis]

        learnable_z = torch.zeros((batch_size,latent_size), device=device, requires_grad=True)
        
        # optimizer
        optimizer_test = optim.Adam([
            {'params':learnable_z, 'lr': 0.01, 'betas':(0.4, .999)}
            ])
        
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
            latent_z = learnable_z/torch.sqrt(torch.sum(learnable_z**2, axis=1))[:,np.newaxis]

            # save loss
            loss_list.append(loss.item())

        # output stats
        # if epoch % 20 == 0:
        print('[{:d}]\tloss: {:.4f}'.format(i, loss))

        # save latent_z for visualization
        recon_img = gen_images.detach().view((1,64,64)).cpu()
        target_image = real_images.view((1,64,64)).cpu()
        img_list.append(vutils.make_grid([recon_img, target_image], padding = 1, normalize = True))
    return img_list, loss_list

def train(num_epochs, name, hot_start = False):
    # dataset
    dataset, dataloader = get_loader(True)

    # networks
    netG =Generator().to(device)
    netG.apply(weights_init)

    # randomlly init z
    latent_z = torch.randn(len(dataset), latent_size, 1, 1, device = device)
    # project to l2
    latent_z /= torch.sqrt(torch.sum(latent_z**2, axis=1))[:,np.newaxis]
    # graded z used for learning
    batch_z = torch.zeros((batch_size,latent_size), device=device, requires_grad=True)

    # optimizer
    optimizer = optim.Adam([
        {'params':netG.parameters(), 'lr': learning_rate, 'betas':(adam_momentum, .999)},
        {'params':batch_z, 'lr': learning_rate, 'betas':(adam_momentum, .999)}
    ])

    # l2 loss
    criterion = nn.MSELoss().to(device)

    # list for stats
    img_list = []
    loss_list = []

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            for i, (batch_data,_, idx) in enumerate(dataloader):
                # extract batch and prepare z for optim
                batch_z.data = latent_z[idx]
                optimizer.zero_grad()

                # generate image batch
                gen_images = netG(batch_z)
                # extract batch
                real_images = batch_data.to(device)
                loss = criterion(real_images.view(-1), gen_images.view(-1))
                # step network and optiizer
                loss.backward()
                optimizer.step()

                # project z and update
                latent_z[idx] = batch_z/torch.sqrt(torch.sum(batch_z**2, axis=1))[:,np.newaxis]

                # output stats
                if i % 100 == 0:
                    # save loss
                    loss_list.append(loss.item())
                    print('[{:d},{:d}]\tloss: {:.4f}'.format(epoch, i, loss))

            if epoch % 5 == 0:
                # save check points
                torch.save({
                    'epoch': epoch + 1,
                    'model': netG.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'latent': latent_z
                }, get_file_path(name))
                    
                # store result of each epoch
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

if __name__ == '__main__':
    parser = arg.ArgumentParser()
    parser.add_argument('--test',action='store_true', help='run test')
    parser.add_argument('name', type=str)
    parser.add_argument('epoch', type=int)
    args = parser.parse_args()
    
    # train!
    if not args.test:
        res_imgs, res_loss = train(args.epoch, args.name)
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
