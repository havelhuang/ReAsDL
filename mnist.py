import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import datasets, transforms
from train.mnist_network import mnist_SimpleMlp

import matplotlib
matplotlib.use('pdf')

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




class mnist:
    def __init__(self, CUDA, op):
        # The bounds in NN-space
        self.x_min = 0
        self.x_max = 1
        self.mean = 0.1307
        self.std = 0.3081
        self.CUDA = CUDA
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        self.model = None
        self.load_model()
        if op == 'before':
            self.load_before_data()
        else:
            self.load_op3_data()


    def load_before_data(self):
        tform1 = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST('./data', train=True, download=True, transform=tform1)


        # stratifily sample from dataset
        targets = np.array(train_data.targets)
        n_class = len(train_data.classes)
        class_rate = 1/n_class * np.ones(n_class)
        n_sample = 5000

        indices = get_data_indices(targets, class_rate, n_sample)

        train_data = Subset(train_data, indices)
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=1000, shuffle=False)

        # Get data into arrays for convenience
        if self.CUDA:
            self.x_train = torch.zeros(len(train_data), 784).cuda()
            self.y_train = torch.zeros(len(train_data), dtype=torch.long).cuda()
            self.y_pred = torch.zeros(len(train_data), dtype=torch.long).cuda()
        else:
            self.x_train = torch.zeros(len(train_data), 784)
            self.y_train = torch.zeros(len(train_data), dtype=torch.long)
            self.y_pred = torch.zeros(len(train_data), dtype=torch.long)


        for idx, (data, target) in enumerate(train_data_loader):
            if self.CUDA:
                data, target = data.float().cuda(), target.long().cuda()
            else:
                data, target = data.float(), target.long()

            data = data.view(-1, 784)
            pred = torch.argmax(self.model(self.data_normalization(data)), dim=1)

            self.x_train[(idx * 1000):((idx + 1) * 1000), :] = data
            self.y_train[(idx * 1000):((idx + 1) * 1000)] = target
            self.y_pred[(idx * 1000):((idx + 1) * 1000)] = pred.detach()


        # contrast_factor = 1.01
        # brighness_factor = 0.05
        # aug_x = torch.clamp(self.x * contrast_factor + brighness_factor, min= 0.0000, max = 1.0000)
        #
        # self.x = torch.cat((self.x,aug_x),0)
        # self.y = torch.cat((self.y,self.y),0)

    def load_op1_data(self):
        tform1 = transforms.Compose([transforms.ToTensor()])
        test_data = datasets.MNIST('./data', train=False, download=True, transform=tform1)

        # stratifily sample from dataset
        targets = np.array(test_data.targets)
        n_class = len(test_data.classes)
        class_rate = 1 / n_class * np.ones(n_class)
        n_sample = 5000

        test_indices = get_data_indices(targets, class_rate, n_sample)

        test_data = Subset(test_data, test_indices)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)

        # Get data into arrays for convenience
        if self.CUDA:
            self.x_test = torch.zeros(len(test_data), 784).cuda()
            self.y_test = torch.zeros(len(test_data), dtype=torch.long).cuda()
            self.y_pred = torch.zeros(len(test_data), dtype=torch.long).cuda()
        else:
            self.x_test = torch.zeros(len(test_data), 784)
            self.y_test = torch.zeros(len(test_data), dtype=torch.long)
            self.y_pred = torch.zeros(len(test_data), dtype=torch.long)

        for idx, (data, target) in enumerate(test_data_loader):
            if self.CUDA:
                data, target = data.float().cuda(), target.long().cuda()
            else:
                data, target = data.float(), target.long()

            data = data.view(-1, 784)
            pred = torch.argmax(self.model(self.data_normalization(data)), dim=1)

            self.x_test[(idx * 1000):((idx + 1) * 1000), :] = data
            self.y_test[(idx * 1000):((idx + 1) * 1000)] = target
            self.y_pred[(idx * 1000):((idx + 1) * 1000)] = pred.detach()

    def load_op2_data(self):
        tform1 = transforms.Compose([transforms.ToTensor()])
        test_data = datasets.MNIST('./data', train=False, download=True, transform=tform1)
        train_data = datasets.MNIST('./data', train=True, download=True, transform=tform1)

        # stratifily sample from dataset
        n_class = len(train_data.classes)
        class_rate = 1 / n_class * np.ones(n_class)
        n_sample_train = 3000
        n_sample_test = 2000

        # concatenate train and test data
        train_indices = get_data_indices(train_data.targets, class_rate, n_sample_train, replacement = True)
        test_indices = get_data_indices(test_data.targets, class_rate, n_sample_test, replacement = True)
        train_data = Subset(train_data, train_indices)
        test_data = Subset(test_data, test_indices)
        test_data = ConcatDataset([train_data, test_data])

        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=True)
        # Get data into arrays for convenience
        if self.CUDA:
            self.x_test = torch.zeros(len(test_data), 784).cuda()
            self.y_test = torch.zeros(len(test_data), dtype=torch.long).cuda()
            self.y_pred = torch.zeros(len(test_data), dtype=torch.long).cuda()
        else:
            self.x_test = torch.zeros(len(test_data), 784)
            self.y_test = torch.zeros(len(test_data), dtype=torch.long)
            self.y_pred = torch.zeros(len(test_data), dtype=torch.long)

        for idx, (data, target) in enumerate(test_data_loader):
            if self.CUDA:
                data, target = data.float().cuda(), target.long().cuda()
            else:
                data, target = data.float(), target.long()

            data = data.view(-1, 784)
            pred = torch.argmax(self.model(self.data_normalization(data)), dim=1)

            self.x_test[(idx * 1000):((idx + 1) * 1000), :] = data
            self.y_test[(idx * 1000):((idx + 1) * 1000)] = target
            self.y_pred[(idx * 1000):((idx + 1) * 1000)] = pred.detach()


    def load_op3_data(self):
        tform1 = transforms.Compose([transforms.ToTensor()])
        test_data = datasets.MNIST('./data', train=False, download=True, transform=tform1)
        train_data = datasets.MNIST('./data', train=True, download=True, transform=tform1)

        # stratifily sample from dataset
        n_class = len(train_data.classes)
        class_rate = 0.08 * np.ones(n_class)
        class_rate[4] = 0.04
        class_rate[2] = 0.20
        class_rate[8] = 0.20
        n_sample_train = 3000
        n_sample_test = 2000

        # concatenate train and test data
        train_indices = get_data_indices(train_data.targets, class_rate, n_sample_train, replacement = True)
        test_indices = get_data_indices(test_data.targets, class_rate, n_sample_test, replacement = True)
        train_data = Subset(train_data, train_indices)
        test_data = Subset(test_data, test_indices)
        test_data = ConcatDataset([train_data, test_data])

        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=True)
        # Get data into arrays for convenience
        if self.CUDA:
            self.x_test = torch.zeros(len(test_data), 784).cuda()
            self.y_test = torch.zeros(len(test_data), dtype=torch.long).cuda()
            self.y_pred = torch.zeros(len(test_data), dtype=torch.long).cuda()
        else:
            self.x_test = torch.zeros(len(test_data), 784)
            self.y_test = torch.zeros(len(test_data), dtype=torch.long)
            self.y_pred = torch.zeros(len(test_data), dtype=torch.long)

        for idx, (data, target) in enumerate(test_data_loader):
            if self.CUDA:
                data, target = data.float().cuda(), target.long().cuda()
            else:
                data, target = data.float(), target.long()

            data = data.view(-1, 784)
            pred = torch.argmax(self.model(self.data_normalization(data)), dim=1)

            self.x_test[(idx * 1000):((idx + 1) * 1000), :] = data
            self.y_test[(idx * 1000):((idx + 1) * 1000)] = target
            self.y_pred[(idx * 1000):((idx + 1) * 1000)] = pred.detach()




    def data_normalization(self, x_input):
        return (x_input - self.mean) / self.std

    def load_model(self):
        # Create model and load trained parameters
        self.model = mnist_SimpleMlp()
        self.model.load_state_dict(torch.load('./data/mnist_simplemlp.pickle'))
        if self.CUDA:
            self.model.cuda()




