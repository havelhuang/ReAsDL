from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from gtsrb_model import Net
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from gtsrb_data import initialize_data, data_transforms,data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast,data_rotate,data_hvflip,data_shear,data_translate,data_center,data_hflip,data_vflip

if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
	use_gpu = False
	print("Using CPU")

FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
Tensor = FloatTensor

data_path = '../data/GTSRB'
epochs = 100
lr = 0.0001
log_interval = 10
batch_size = 64

initialize_data(data_path)

# Apply data transformations on the training images to augment dataset
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([datasets.ImageFolder(data_path + '/trainingset',
    transform=data_transforms), datasets.ImageFolder(data_path + '/trainingset',
    transform=data_jitter_brightness), datasets.ImageFolder(data_path + '/trainingset',
    transform=data_jitter_hue), datasets.ImageFolder(data_path + '/trainingset',
    transform=data_jitter_contrast), datasets.ImageFolder(data_path + '/trainingset',
    transform=data_jitter_saturation), datasets.ImageFolder(data_path + '/trainingset',
    transform=data_translate), datasets.ImageFolder(data_path + '/trainingset',
    transform=data_rotate), datasets.ImageFolder(data_path + '/trainingset',
    transform=data_hvflip), datasets.ImageFolder(data_path + '/trainingset',
    transform=data_center), datasets.ImageFolder(data_path + '/trainingset',
    transform=data_shear)]), batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=use_gpu)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(data_path + '/valset',
    transform=data_transforms), batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=use_gpu)


# Neural Network and Optimizer
model = Net()


if use_gpu:
    model.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr = lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.5,verbose=True)

def train(epoch):
    model.train()
    correct = 0
    training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if use_gpu:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        max_index = output.max(dim = 1)[1]
        correct += (max_index == target).sum()
        training_loss += loss
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss per example: {:.6f}\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data.item()/(batch_size * log_interval),loss.data.item()))
    print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                training_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            if use_gpu:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            validation_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    scheduler.step(np.around(validation_loss,2))
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


for epoch in range(1, epochs + 1):
    train(epoch)
    validation()
    model_file = 'gtsrb_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. Run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')
