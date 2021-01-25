# MIT License
#
# Copyright (c) 2018, University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

CUDA = True

import os
import time
import pickle
import numpy as np
if CUDA:
  cuda_id = '0'
  os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
  os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(cuda_id)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from train.cifar10_models import *
import utils

def cm2inch(value):
  return value/2.54

# Fixing random seed for reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def run():
  # Load data
  #kwargs = {'num_workers': 1, 'pin_memory': True}
  train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=128, shuffle=True, num_workers=2)
  test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=100, shuffle=False, num_workers=2)

  # Create model
  model = SimpleDLA()
  if CUDA:
    model.cuda()

  # Get an optimizer
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
  loss_fn = torch.nn.CrossEntropyLoss()

  # Create the function to perform an epoch
  def train(epoch):
    model.train()
    total_loss = 0
    for _, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      if CUDA:
        data, target = data.float().cuda(), target.long().cuda()
      else:
        data, target = data.float(), target.long()

      #print(data.size())
      #raise Exception()

      # data = data.view(-1, 32*32*3)

      #print(data.size(), target.size())
    
      output = model(data)

      #print(output.size())
      #raise Exception()

      loss = loss_fn(output, target)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

    total_loss /= len(train_loader.dataset)
    return total_loss

  def test(epoch):
    model.eval()
    total_loss = 0
    total_correct = 0
    for _, (data, target) in enumerate(test_loader):
      if CUDA:
        data, target = data.float().cuda(), target.long().cuda()
      else:
        data, target = data.float(), target.long()
      # data = data.view(-1, 32*32*3)
    
      output = model(data)
      preds = output.argmax(dim=1)
      total_correct += (preds == target).sum().float().item()
      loss = loss_fn(output, target)
      total_loss += loss.item()

    total_loss /= len(test_loader.dataset)
    total_correct /= len(test_loader.dataset)
    return total_loss, total_correct

  # Train the network
  print("Starting network training")
  for epoch in range(1, 50):
      train_loss = train(epoch)
      with torch.no_grad():
        test_loss, test_acc = test(epoch)

      print("[epoch %03d]  train loss: %.5f, test loss: %.5f, test acc: %.3f" % (epoch, train_loss, test_loss, test_acc))

  print("Training Done.")
  print(f"Final Loss: {test_loss}")

  torch.save(model.state_dict(), '../data/cifar10_simplemlp.pickle')

run()