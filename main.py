
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

CUDA = True

from copy import deepcopy
import sys
from mnist import mnist
from cifar10 import cifar10
from op_learning import op_learning
import multiprocessing
from sklearn.neighbors import KernelDensity

import math
import os
import time
import pickle
import numpy as np
if CUDA:
  cuda_id = '0'
  os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
  os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(cuda_id)
import torch
from multi_level import multilevel_uniform, greyscale_multilevel_uniform
import torch.distributions as dist
import torchvision
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

plt.style.use(['seaborn-white', 'seaborn-paper', 'seaborn-ticks'])
matplotlib.rc('font', family='Latin Modern Roman')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# from plnn.model import load_and_simplify2
import utils

def cm2inch(value):
  return value/2.54

def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
  with multiprocessing.Pool(thread_count) as p:
    return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))


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
  else:
    print('please choose a model from mnist, cifar10!')
    sys.exit(1)

  if op == 'before':
    x_op = loader.x
    y_op = loader.y
    x_latent = loader.x_latent
    print('Prior to the operational testing, running with the existing data.')
  else:
    raise Exception("Please define an Operational Dataset")


  # r-separation to decide cell size
  # nns, ret = utils.get_nearest_oppo_dist(np.array(x_latent.cpu()), np.array(y_op.cpu()), np.inf, n_jobs=10)
  # ret = np.sort(ret)
  # print(ret.min(), ret.mean())

  # latent space check the latent variables' value range
  # max_x = torch.amax(x_latent, 0)
  # min_x = torch.amin(x_latent, 0)

  input_learn = op_learning(y_op, 5.5, -4.5, cell_size)
  input_learn.init_op(x_latent)
  new_x = input_learn.kde.sample(1000, random_state=0)

  device = torch.device("cuda:0" if CUDA else "cpu")
  
  new_x = torch.tensor(new_x, device=device).float()
  input_learn.updata_op(new_x, None)


  cell_points = input_learn.unique_symbs * input_learn.cell_interval + input_learn.x_min
  op_points = parrallel_score_samples(input_learn.kde, np.array(cell_points.cpu()))
  op_points = np.exp(op_points)
  
  sort_id = np.argsort(-op_points)
  op_points = -np.sort(-op_points)
  cell_points = cell_points[sort_id]
  input_learn.unique_symbs = input_learn.unique_symbs[sort_id]
  input_learn.find_ground_truth()
  np.save('op_cell.npy', op_points)


  # op_points = np.sort(op_points)[-100000:]
  cell_volume = math.pow(10/cell_size, 8)
  op_model =  sum(op_points*cell_volume)


  input_points = input_learn.cal_pred_label(loader,cell_points)

  # torchvision.utils.save_image(input_points[100:120], 'output/real_samples.png')

  # set parameters for multi-level splitting
  v = 2
  rho = 0.1
  debug= True
  stats=True
  sigma = 0.3

  print('rho', rho, 'count_particles', count_particles, 'count_mh_steps', count_mh_steps)

  # create empty list to save the sampling results
  cell_lambda = []
  max_vals = []
  levels = []

  # total samples n and failure observations k
  sample_count = 0
  sample_fail = 0

  # verify the probability of failure for each cell
  for idx in range(len(input_points)):
    print('--------------------------------')
    sample_count += 1
    x_class = input_learn.unique_y[idx]
    x_sample = input_points[idx]
    print(f'cell {idx}, label {x_class}')

    if x_class == 'cross':
      cell_lambda.append(1)
      print('cross-boundary cell, conservatively set pfd = 1')
      continue

    elif x_class == 'empty':
      x_class = input_learn.unique_y_pred[idx]

    if x_class != input_learn.unique_y_pred[idx]:
      cell_lambda.append(1)
      continue


    def prop(x_input):
      x_input = loader.data_normalization(x_input)
      y_pred = loader.model(x_input)
      y_diff = torch.cat((y_pred[:, :x_class], y_pred[:, (x_class + 1):]), dim=1) - y_pred[:, x_class].unsqueeze(-1)
      y_diff, _ = y_diff.max(dim=1)
      return y_diff  # .max(dim=1)

    start = time.time()
    with torch.no_grad():
      lg_p, max_val, _, l = robust_test(prop, x_sample, sigma, CUDA=CUDA, rho=rho, count_particles=count_particles,
                                             count_mh_steps=count_mh_steps, debug=debug, stats=stats)
    end = time.time()
    print(f'Took {(end - start) / 60} minutes...')


    cell_lambda.append(10 ** (lg_p))
    max_vals.append(max_val)
    levels.append(l)

    if idx % 2000 == 0:
      np.save('pfd_cell.npy', np.array(cell_lambda))

    if debug:
      print('lg_p', lg_p, 'max_val', max_val)

  np.save('pfd_cell.npy', np.array(cell_lambda))


if __name__ == "__main__":
    main('mnist', 'before', cell_size = 100, count_mh_steps = 200, count_particles = 1000)








