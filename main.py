
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

CUDA = False


import sys
from mnist import mnist
from cifar10 import cifar10
from gtsrb import gtsrb

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
from multi_level import multilevel_uniform, greyscale_multilevel_uniform

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

plt.style.use(['seaborn-white', 'seaborn-paper', 'seaborn-ticks'])
matplotlib.rc('font', family='Latin Modern Roman')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from plnn.model import load_and_simplify2
import utils

def cm2inch(value):
  return value/2.54


def main(test_model, op, cell_size = 3, count_mh_steps = 100, count_particles = 1000):
  # Fix random seed for reproducibility

  seed = 5
  np.random.seed(seed)
  torch.manual_seed(seed)

  # load MNIST class
  # load cifar10 class
  # load gtsrb class
  if test_model == 'mnist':
    loader = mnist(CUDA,op)
    robust_test = greyscale_multilevel_uniform
  elif test_model == 'cifar10':
    loader = cifar10(CUDA,op)
    robust_test = multilevel_uniform
  elif test_model == 'gtsrb':
    loader = gtsrb(CUDA,op)
    robust_test = multilevel_uniform
  else:
    print('please choose a model from mnist, cifar10, and gtsrb!')
    sys.exit(1)


  if op == 'op':
    fname = 'output'
    file = open("data/cell_symb.pkl", 'rb')
    cell_symb = pickle.load(file)
    file.close()
    bef_sample = np.load('data/train_sample_number.npy')
    bef_sample_count = bef_sample[1]
    bef_sample_fail = bef_sample[0]
    x_op = loader.x_test
    y_op = loader.y_test
    print('During the operational testing.\n')

  elif op == 'before':
    fname = 'data'
    cell_symb = {}
    bef_sample_count = 0
    bef_sample_fail = 0
    x_op = loader.x_train
    y_op = loader.y_train
    print('Prior to the operational testing, running with the existing data.')

  r = utils.record('output/record.txt', time.time())

  # find sybolic representation of unique cells via current data
  bins = np.linspace(loader.x_min, loader.x_max, num=cell_size + 1)
  bins2 = bins[1:len(bins) - 1]

  if CUDA:
    symbs = np.digitize(np.array(x_op.cpu()), bins2)
  else:
    symbs = np.digitize(np.array(x_op), bins2)

  ######################################################
  unique_symbs, unique_indices, unique_counts = np.unique(symbs, axis=0, return_counts=True, return_index=True)
  # unique_class = y[unique_indices]
  # aa = model(x[1])
  # aaa = unique_symbs[1]
  # aaaa = np.resize(aaa,(28,28))
  # print(len(unique_counts))
  ######################################################


  # set parameters for multi-level splitting
  v = 2
  rho = 0.1
  debug= True
  stats=True

  print('rho', rho, 'count_particles', count_particles, 'count_mh_steps', count_mh_steps)

  # create empty list to save the sampling results
  lg_ps = []
  max_vals = []
  levels = []

  # total samples n and failure observations k
  sample_count = 0
  sample_fail = 0

  # verify the probability of failure for each cell
  for idx in range(len(symbs)):
    print('--------------------------------')

    sample_count += 1
    symb = symbs[idx]
    symb_rep = symb.flatten()
    symb_rep = hash(tuple(symb_rep))
    symb_rep = repr(symb_rep)

    x_class = y_op[idx]
    if x_class != loader.y_pred[idx]:
        sample_fail += 1


    if symb_rep in cell_symb.keys():
      value = cell_symb[symb_rep]
      if y_op[idx] != value[0]:
        value[1] = 0
        print('cross-boundary cell, conservatively set pfd = 1')
      else:
        print('existing cell, no need to update')
      value[-1] += 1
    else:
      # calculate the range of cell


      x_low = np.float32(bins[symb])
      x_high = np.float32(bins[symb + 1])

      print(f'cell {idx}, label {x_class}')


      def prop(x_input):
        x_input = loader.data_normalization(x_input)
        y_pred = loader.model(x_input)
        y_diff = torch.cat((y_pred[:, :x_class], y_pred[:, (x_class + 1):]), dim=1) - y_pred[:, x_class].unsqueeze(-1)
        y_diff, _ = y_diff.max(dim=1)
        return y_diff  # .max(dim=1)


      start = time.time()
      with torch.no_grad():
        lg_p, max_val, _, l = robust_test(prop, x_low, x_high, cell_size, CUDA=CUDA, rho=rho, count_particles=count_particles,
                                               count_mh_steps=count_mh_steps, debug=debug, stats=stats)
      end = time.time()
      print(f'Took {(end - start) / 60} minutes...')

      cell_symb[symb_rep] = [x_class, lg_p, 1]

      lg_ps.append(lg_p)
      max_vals.append(max_val)
      levels.append(l)

      if debug:
        print('lg_p', lg_p, 'max_val', max_val)

    # output the updated probability of failure of model
    pfd = utils.model_pfd(cell_symb, sample_count + bef_sample_count, v)

    # output the average failure of model
    avg_fail = utils.model_avg_fail(cell_symb)

    # output the MLE failure of model
    mle_fail = (sample_fail + bef_sample_fail)/(sample_count + bef_sample_count)
    print('MLE failure estimation:', mle_fail)

    # write to the file
    utils.writeInfo(r, idx, pfd, avg_fail, mle_fail)

    if sample_count % 2000 == 0:
      f = open("output/cell_symb.pkl", "wb")
      pickle.dump(cell_symb, f)
      f.close()
      np.save('output/train_sample_number.npy', np.array([sample_fail, sample_count]))


  r.close()

  f = open(fname + "/cell_symb.pkl","wb")
  pickle.dump(cell_symb,f)
  f.close()

  np.save(fname + '/train_sample_number.npy', np.array([sample_fail, sample_count]))

if __name__ == "__main__":
    main('cifar10', 'before', cell_size = 20, count_mh_steps = 100, count_particles = 500)








