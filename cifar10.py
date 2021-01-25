import os
import sys
#sys.path.append('train/LaNet')
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import datasets, transforms
from train.cifar10_models import *
#from train.LaNet.model import NetworkCIFAR as Network
#from train.LaNet.nasnet_set import *

from random import seed
import random
seed(1)

def get_data_indices(targets, class_rate, n_sample, replacement = None):
    indices = []
    class_n = np.rint(n_sample * class_rate)

    if replacement:
        b = 7
        sample_list = [1,2,3,4,5,6,7]
        weight = [40, 25, 15, 10, 5, 2.5, 2.5]
        for i, value in enumerate(np.array(targets)):
            if class_n[value] > b:
                n = random.choices(sample_list, weights = weight, k=1)
                class_n[value] -= n[0]
                indices += [i] * n[0]

            elif class_n[value] > 0 and class_n[value] <= b:
                class_n[value] -= 1
                indices += [i]

            if sum(class_n) == 0:
                break

    else:
        for i, value in enumerate(np.array(targets)):
            if class_n[value] > 0:
                class_n[value] -= 1
                indices += [i]

            if sum(class_n) == 0:
                break

    return np.array(indices)





class cifar10:
    def __init__(self, CUDA, op):
        # The bounds in NN-space
        self.x_min = 0
        self.x_max = 1
        self.cifar_mean = [0.4914, 0.4822, 0.4465]
        self.cifar_std = [0.2023, 0.1994, 0.2010]
        self.CUDA = CUDA
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.model = None
        self.load_model()
        if op == 'before':
            self.load_before_data()
        else:
            self.load_op3_data()


    def load_before_data(self):
        tform1 = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=tform1)


        # stratifily sample from dataset
        targets = np.array(train_data.targets)
        n_class = len(train_data.classes)
        class_rate = 1 / n_class * np.ones(n_class)
        n_sample = 5000

        indices = get_data_indices(targets, class_rate, n_sample)

        train_data = Subset(train_data, indices)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)


        # Get data into arrays for convenience
        if self.CUDA:
            self.x_train = torch.zeros(len(train_data), 3, 32, 32).cuda()
            self.y_train = torch.zeros(len(train_data), dtype=torch.long).cuda()
            self.y_pred = torch.zeros(len(train_data), dtype=torch.long).cuda()
        else:
            self.x_train = torch.zeros(len(train_data), 3, 32, 32)
            self.y_train = torch.zeros(len(train_data), dtype=torch.long)
            self.y_pred = torch.zeros(len(train_data), dtype=torch.long)

        for idx, (data, target) in enumerate(train_loader):
            if self.CUDA:
                data, target = data.float().cuda(), target.long().cuda()
            else:
                data, target = data.float(), target.long()

            pred = torch.argmax(self.model(self.data_normalization(data)), dim=1)

            # data = data.view(-1, 3*32*32)
            self.x_train[(idx * 100):((idx + 1) * 100), :, :, :] = data
            self.y_train[(idx * 100):((idx + 1) * 100)] = target
            self.y_pred[(idx * 100):((idx + 1) * 100)] = pred.detach()


    def load_op1_data(self):
        tform1 = transforms.Compose([transforms.ToTensor()])
        test_data = datasets.CIFAR10('./data', train=False, download=True, transform=tform1)

        # stratifily sample from dataset
        targets = np.array(test_data.targets)
        n_class = len(test_data.classes)
        class_rate = 1 / n_class * np.ones(n_class)
        n_sample = 5000

        indices = get_data_indices(targets, class_rate, n_sample)

        test_data = Subset(test_data, indices)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)

        # Get data into arrays for convenience
        if self.CUDA:
            self.x_test = torch.zeros(len(test_data), 3, 32, 32).cuda()
            self.y_test = torch.zeros(len(test_data), dtype=torch.long).cuda()
            self.y_pred = torch.zeros(len(test_data), dtype=torch.long).cuda()
        else:
            self.x_test = torch.zeros(len(test_data), 3, 32, 32)
            self.y_test = torch.zeros(len(test_data), dtype=torch.long)
            self.y_pred = torch.zeros(len(test_data), dtype=torch.long)

        for idx, (data, target) in enumerate(test_loader):
            if self.CUDA:
                data, target = data.float().cuda(), target.long().cuda()
            else:
                data, target = data.float(), target.long()

            pred = torch.argmax(self.model(self.data_normalization(data)), dim=1)

            # data = data.view(-1, 3*32*32)
            self.x_test[(idx * 100):((idx + 1) * 100), :, :, :] = data
            self.y_test[(idx * 100):((idx + 1) * 100)] = target
            self.y_pred[(idx * 100):((idx + 1) * 100)] = pred.detach()

    def load_op2_data(self):
        tform1 = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=tform1)
        test_data = datasets.CIFAR10('./data', train=False, download=True, transform=tform1)

        # stratifily sample from dataset
        n_class = len(train_data.classes)
        class_rate = 1 / n_class * np.ones(n_class)
        n_sample_train = 3000
        n_sample_test = 2000

        # concatenate train and test data
        train_indices = get_data_indices(train_data.targets, class_rate, n_sample_train, replacement=True)
        test_indices = get_data_indices(test_data.targets, class_rate, n_sample_test, replacement=True)
        train_data = Subset(train_data, train_indices)
        test_data = Subset(test_data, test_indices)
        test_data = ConcatDataset([train_data, test_data])

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)

        # Get data into arrays for convenience
        if self.CUDA:
            self.x_test = torch.zeros(len(test_data), 3, 32, 32).cuda()
            self.y_test = torch.zeros(len(test_data), dtype=torch.long).cuda()
            self.y_pred = torch.zeros(len(test_data), dtype=torch.long).cuda()
        else:
            self.x_test = torch.zeros(len(test_data), 3, 32, 32)
            self.y_test = torch.zeros(len(test_data), dtype=torch.long)
            self.y_pred = torch.zeros(len(test_data), dtype=torch.long)

        for idx, (data, target) in enumerate(test_loader):
            if self.CUDA:
                data, target = data.float().cuda(), target.long().cuda()
            else:
                data, target = data.float(), target.long()

            pred = torch.argmax(self.model(self.data_normalization(data)), dim=1)

            # data = data.view(-1, 3*32*32)
            self.x_test[(idx * 100):((idx + 1) * 100), :, :, :] = data
            self.y_test[(idx * 100):((idx + 1) * 100)] = target
            self.y_pred[(idx * 100):((idx + 1) * 100)] = pred.detach()

    def load_op3_data(self):
        tform1 = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=tform1)
        test_data = datasets.CIFAR10('./data', train=False, download=True, transform=tform1)

        # stratifily sample from dataset
        n_class = len(train_data.classes)
        class_rate = 0.08 * np.ones(n_class)
        class_rate[1] = 0.10
        class_rate[3] = 0.10
        class_rate[7] = 0.16
        class_rate[5] = 0.16
        n_sample_train = 3000
        n_sample_test = 2000

        # concatenate train and test data
        train_indices = get_data_indices(train_data.targets, class_rate, n_sample_train, replacement=True)
        test_indices = get_data_indices(test_data.targets, class_rate, n_sample_test, replacement=True)
        train_data = Subset(train_data, train_indices)
        test_data = Subset(test_data, test_indices)
        test_data = ConcatDataset([train_data, test_data])

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)

        # Get data into arrays for convenience
        if self.CUDA:
            self.x_test = torch.zeros(len(test_data), 3, 32, 32).cuda()
            self.y_test = torch.zeros(len(test_data), dtype=torch.long).cuda()
            self.y_pred = torch.zeros(len(test_data), dtype=torch.long).cuda()
        else:
            self.x_test = torch.zeros(len(test_data), 3, 32, 32)
            self.y_test = torch.zeros(len(test_data), dtype=torch.long)
            self.y_pred = torch.zeros(len(test_data), dtype=torch.long)

        for idx, (data, target) in enumerate(test_loader):
            if self.CUDA:
                data, target = data.float().cuda(), target.long().cuda()
            else:
                data, target = data.float(), target.long()

            pred = torch.argmax(self.model(self.data_normalization(data)), dim=1)

            # data = data.view(-1, 3*32*32)
            self.x_test[(idx * 100):((idx + 1) * 100), :, :, :] = data
            self.y_test[(idx * 100):((idx + 1) * 100)] = target
            self.y_pred[(idx * 100):((idx + 1) * 100)] = pred.detach()


    def data_normalization(self, x_input):
        transform = transforms.Compose([transforms.Normalize(mean = self.cifar_mean, std = self.cifar_std)])
        return transform(x_input)



    def load_model(self):
        # Create model and load trained parameters
        # arch = '[2, 2, 0, 2, 1, 2, 0, 2, 2, 3, 2, 1, 2, 0, 0, 1, 1, 1, 2, 1, 1, 0, 3, 4, 3, 0, 3, 1]'
        # net = eval(arch)
        # code = gen_code_from_list(net, node_num=int((len(net) / 4)))
        # genotype = translator([code, code], max_node=int((len(net) / 4)))
        # self.model = Network(128, 10, 24, True, genotype).cuda()
        # checkpoint = torch.load('./data/lanas_128_99.03' + '/top1.pt')
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = SimpleDLA()
        self.model.load_state_dict(torch.load('./data/cifar10_simpledla.pickle', map_location=torch.device('cpu')))
        self.model.eval()
        if self.CUDA:
            self.model.cuda()












