from __future__ import print_function

import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import sys
from PIL import Image

import torchvision.transforms as transforms

# data augmentation for training and test time
# Resize all images to 32 * 32 and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from the training set

data_transforms = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image brightness
data_jitter_brightness = transforms.Compose([
	transforms.Resize((32, 32)),
    # transforms.ColorJitter(brightness=-5),
    transforms.ColorJitter(brightness=5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image saturation
data_jitter_saturation = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.ColorJitter(saturation=5),
    # transforms.ColorJitter(saturation=-5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image contrast
data_jitter_contrast = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.ColorJitter(contrast=5),
    # transforms.ColorJitter(contrast=-5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image hues
data_jitter_hue = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.ColorJitter(hue=0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and rotate image
data_rotate = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image horizontally and vertically
data_hvflip = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(1),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image horizontally
data_hflip = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image vertically
data_vflip = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and shear image
data_shear = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomAffine(degrees = 15,shear=2),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and translate image
data_translate = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomAffine(degrees = 15,translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and crop image 
data_center = transforms.Compose([
	transforms.Resize((36, 36)),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and convert image to grayscale
data_grayscale = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])


def initialize_data(folder):

    # locate train_data
    train_folder = folder + '/trainingset'
    if not os.path.isdir(train_folder):
        print(train_folder + ' not found')

        
    # locate test_data
    test_folder = folder + '/testset'
    if not os.path.isdir(test_folder):
        print(test_folder + 'not found')

        
    # make validation_data by using images 00000*, 00001* and 00002* in each class
    val_folder = folder + '/valset'
    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)
        for dirs in os.listdir(train_folder):
            if dirs.startswith('000'):
                os.mkdir(val_folder + '/' + dirs)
                for f in os.listdir(train_folder + '/' + dirs):
                    if f.startswith('00000') or f.startswith('00001') or f.startswith('00002'):
                        # move file to validation folder
                        os.rename(train_folder + '/' + dirs + '/' + f, val_folder + '/' + dirs + '/' + f)


class GTSRB(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root_dir, train=False, transform=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        if not os.path.isdir(os.path.join(self.root_dir, self.base_folder)):
            print('please download GTSRB from https://1drv.ms/u/s!An8jrZtDgrMljdt7o2khe7TGmZWbUg, unzip file and move to ./data folder.')
            sys.exit(1)


        self.sub_directory = 'trainingset' if train else 'testset'
        self.csv_file_name = 'training.csv' if train else 'test.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[idx, 0])
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId
