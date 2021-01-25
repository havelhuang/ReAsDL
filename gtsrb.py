import train.gtsrb_data as dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from train.gtsrb_model import Net
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import transforms
from train.cifar10_models import *
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



class gtsrb:
    def __init__(self,CUDA,op):
        # The bounds in NN-space
        self.x_min = 0
        self.x_max = 1
        self.gtsrb_mean = [0.3337, 0.3064, 0.3171]
        self.gtsrb_std = [0.2672, 0.2564, 0.2629]
        self.CUDA = CUDA
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_pred = None
        # self.classes = None
        self.model = None
        self.load_model()
        if op == 'before':
            self.load_before_data()
        else:
            self.load_op3_data()



    def load_before_data(self):
        tform1 = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        # Create Datasets
        trainset = dataset.GTSRB(root_dir='./data', train=True, transform = tform1)


        # stratifily sample from dataset
        targets = np.array(trainset.csv_data)[:,1]
        n_class = 43
        class_rate = 1 / n_class * np.ones(n_class)
        n_sample = 4300
        indices = get_data_indices(targets, class_rate, n_sample)
        trainset = Subset(trainset, indices)

        # Load Datasets
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle = True)


        # Get train data into arrays for convenience
        if self.CUDA:
            self.x_train = torch.zeros(len(trainset), 3, 32, 32).cuda()
            self.y_train = torch.zeros(len(trainset), dtype=torch.long).cuda()
            self.y_pred = torch.zeros(len(trainset), dtype=torch.long).cuda()
        else:
            self.x_train = torch.zeros(len(trainset), 3, 32, 32)
            self.y_train = torch.zeros(len(trainset), dtype=torch.long)
            self.y_pred = torch.zeros(len(trainset), dtype=torch.long)


        for idx, (data, target) in enumerate(train_loader):
            if self.CUDA:
                data, target = data.float().cuda(), target.long().cuda()
            else:
                data, target = data.float(), target.long()

            pred = torch.argmax(self.model(self.data_normalization(data)), dim=1)

            # data = data.view(-1, 3*32*32)
            self.x_train[(idx * 1000):((idx + 1) * 1000), :, :, :] = data
            self.y_train[(idx * 1000):((idx + 1) * 1000)] = target
            self.y_pred[(idx * 1000):((idx + 1) * 1000)] = pred.detach()


    def load_op1_data(self):
        tform1 = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        # Create Datasets
        testset = dataset.GTSRB(root_dir='./data', train=False, transform = tform1)

        # stratifily sample from dataset
        targets = np.array(testset.csv_data)[:, 1]
        n_class = 43
        class_rate = 1 / n_class * np.ones(n_class)
        n_sample = 4300
        indices = get_data_indices(targets, class_rate, n_sample)
        testset = Subset(testset, indices)

        # Load Datasets
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle = True)


        # Get train data into arrays for convenience
        if self.CUDA:
            self.y_pred = torch.zeros(len(testset), dtype=torch.long).cuda()
            self.x_test = torch.zeros(len(testset), 3, 32, 32).cuda()
            self.y_test = torch.zeros(len(testset), dtype=torch.long).cuda()
        else:
            self.y_pred = torch.zeros(len(testset), dtype=torch.long)
            self.x_test = torch.zeros(len(testset), 3, 32, 32)
            self.y_test = torch.zeros(len(testset), dtype=torch.long)


        for idx, (data, target) in enumerate(test_loader):
            if self.CUDA:
                data, target = data.float().cuda(), target.long().cuda()
            else:
                data, target = data.float(), target.long()

            pred = torch.argmax(self.model(self.data_normalization(data)), dim=1)

            # data = data.view(-1, 3*32*32)
            self.x_test[(idx * 1000):((idx + 1) * 1000), :, :, :] = data
            self.y_test[(idx * 1000):((idx + 1) * 1000)] = target
            self.y_pred[(idx * 1000):((idx + 1) * 1000)] = pred.detach()

    def load_op2_data(self):
        tform1 = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        # Create Datasets
        trainset = dataset.GTSRB(root_dir='./data', train=True, transform = tform1)
        testset = dataset.GTSRB(root_dir='./data', train=False, transform = tform1)

        # stratifily sample from dataset
        n_class = 43
        class_rate = 1 / n_class * np.ones(n_class)
        n_sample_train = 2150
        n_sample_test = 2150

        # concatenate train and test data
        train_indices = get_data_indices(np.array(trainset.csv_data)[:, 1], class_rate, n_sample_train, replacement=True)
        test_indices = get_data_indices(np.array(testset.csv_data)[:, 1], class_rate, n_sample_test, replacement=True)
        trainset = Subset(trainset, train_indices)
        testset = Subset(testset, test_indices)
        testset = ConcatDataset([trainset, testset])

        # Load Datasets
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle = True)

        # Get train data into arrays for convenience
        if self.CUDA:
            self.y_pred = torch.zeros(len(testset), dtype=torch.long).cuda()
            self.x_test = torch.zeros(len(testset), 3, 32, 32).cuda()
            self.y_test = torch.zeros(len(testset), dtype=torch.long).cuda()
        else:
            self.y_pred = torch.zeros(len(testset), dtype=torch.long)
            self.x_test = torch.zeros(len(testset), 3, 32, 32)
            self.y_test = torch.zeros(len(testset), dtype=torch.long)

        for idx, (data, target) in enumerate(test_loader):
            if self.CUDA:
                data, target = data.float().cuda(), target.long().cuda()
            else:
                data, target = data.float(), target.long()

            pred = torch.argmax(self.model(self.data_normalization(data)), dim=1)

            # data = data.view(-1, 3*32*32)
            self.x_test[(idx * 1000):((idx + 1) * 1000), :, :, :] = data
            self.y_test[(idx * 1000):((idx + 1) * 1000)] = target
            self.y_pred[(idx * 1000):((idx + 1) * 1000)] = pred.detach()

    def load_op3_data(self):
        tform1 = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        # Create Datasets
        trainset = dataset.GTSRB(root_dir='./data', train=True, transform = tform1)
        testset = dataset.GTSRB(root_dir='./data', train=False, transform = tform1)

        # stratifily sample from dataset
        n_class = 43
        class_rate = 0.02 * np.ones(n_class)
        class_rate[15] = 0.09
        class_rate[32] = 0.09
        n_sample_train = 2150
        n_sample_test = 2150

        # concatenate train and test data
        train_indices = get_data_indices(np.array(trainset.csv_data)[:, 1], class_rate, n_sample_train, replacement=True)
        test_indices = get_data_indices(np.array(testset.csv_data)[:, 1], class_rate, n_sample_test, replacement=True)
        trainset = Subset(trainset, train_indices)
        testset = Subset(testset, test_indices)
        testset = ConcatDataset([trainset, testset])

        # Load Datasets
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle = True)

        # Get train data into arrays for convenience
        if self.CUDA:
            self.y_pred = torch.zeros(len(testset), dtype=torch.long).cuda()
            self.x_test = torch.zeros(len(testset), 3, 32, 32).cuda()
            self.y_test = torch.zeros(len(testset), dtype=torch.long).cuda()
        else:
            self.y_pred = torch.zeros(len(testset), dtype=torch.long)
            self.x_test = torch.zeros(len(testset), 3, 32, 32)
            self.y_test = torch.zeros(len(testset), dtype=torch.long)

        for idx, (data, target) in enumerate(test_loader):
            if self.CUDA:
                data, target = data.float().cuda(), target.long().cuda()
            else:
                data, target = data.float(), target.long()

            pred = torch.argmax(self.model(self.data_normalization(data)), dim=1)

            # data = data.view(-1, 3*32*32)
            self.x_test[(idx * 1000):((idx + 1) * 1000), :, :, :] = data
            self.y_test[(idx * 1000):((idx + 1) * 1000)] = target
            self.y_pred[(idx * 1000):((idx + 1) * 1000)] = pred.detach()

    def data_normalization(self, x_input):
        return transforms.functional.normalize(x_input, self.gtsrb_mean, self.gtsrb_std)


    def load_model(self):
        # Create model and load trained parameters
        self.model = Net()
        self.model.load_state_dict(torch.load('./data/gtsrb_13.pth', map_location=torch.device('cpu')))
        self.model.eval()
        if self.CUDA:
            self.model.cuda()


