import os
import sys
#sys.path.append('train/LaNet')
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import datasets, transforms
from train.cifar10_models import *
from model import VAE
import utils
from sklearn.metrics import accuracy_score
#from train.LaNet.model import NetworkCIFAR as Network
#from train.LaNet.nasnet_set import *

from random import seed
import random
seed(1)


class cifar10:
    def __init__(self, CUDA, op):
        # The bounds in NN-space
        self.x_min = 0
        self.x_max = 1
        self.cifar_mean = [0.4914, 0.4822, 0.4465]
        self.cifar_std = [0.2023, 0.1994, 0.2010]
        self.CUDA = CUDA
        self.z_size = 16
        self.batch_size = 128
        self.x = None
        self.y = None
        self.x_latent = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        self.model = None
        self.g_model = None
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.load_model()
        if op == 'before':
            self.load_before_data()



    def load_before_data(self):
        tform1 = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=tform1)
        test_data = datasets.CIFAR10('./data', train=False, download=True, transform=tform1)
        op_data = ConcatDataset([train_data, test_data])
        data_loader = torch.utils.data.DataLoader(op_data, batch_size=self.batch_size, shuffle=True)

        # Get data into arrays for convenience
        if self.CUDA:
            self.x = torch.zeros(len(op_data), 3, 32, 32).cuda()
            self.y = torch.zeros(len(op_data), dtype=torch.long).cuda()
            self.x_latent = torch.zeros(len(op_data), self.z_size).cuda()
            self.y_pred = torch.zeros(len(op_data), dtype=torch.long).cuda()
        else:
            self.x = torch.zeros(len(op_data), 3, 32, 32)
            self.y = torch.zeros(len(op_data), dtype=torch.long)
            self.x_latent = torch.zeros(len(op_data), self.z_size)
            self.y_pred = torch.zeros(len(op_data), dtype=torch.long)

        with torch.no_grad():
            for idx, (data, target) in enumerate(data_loader):
                if self.CUDA:
                    data, target = data.float().cuda(), target.long().cuda()
                else:
                    data, target = data.float(), target.long()

                (mean, logvar), x_reconstructed = self.g_model.forward(data)
                pred = torch.argmax(self.model(self.data_normalization(data)), dim=1)

                self.x_latent[(idx * self.batch_size):((idx + 1) * self.batch_size), :] = mean
                self.x[(idx * self.batch_size):((idx + 1) * self.batch_size), :, :, :] = data
                self.y[(idx * self.batch_size):((idx + 1) * self.batch_size)] = target
                self.y_pred[(idx * self.batch_size):((idx + 1) * self.batch_size)] = pred.detach()

        # print("Training set score: %f" % accuracy_score(np.array(self.y.cpu()), np.array(self.y_pred.cpu())))





    def data_normalization(self, x_input):
        transform = transforms.Compose([transforms.Normalize(mean = self.cifar_mean, std = self.cifar_std)])
        return transform(x_input)



    def load_model(self):
        self.model = SimpleDLA()
        self.model.load_state_dict(torch.load('./data/cifar10_simpledla.pickle'))
        self.model.eval()
        if self.CUDA:
            self.model.cuda()

        self.g_model = VAE(
            label='cifar10',
            image_size=32,
            channel_num=3,
            kernel_num=32,
            z_size=self.z_size,
        )

        _ = utils.load_checkpoint(self.g_model, './data')
        self.g_model.eval()

        if self.CUDA:
            self.model.cuda()
            self.g_model.cuda()












