
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os, uuid, logging, math
import torch
import numpy as np
import matplotlib as plt
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

# Orders vertices so they go clockwise or anti-clockwise around polygon
def order_vertices(vertices):
  center = vertices.mean(axis=0)
  angles = np.arctan2(vertices[:,1] - center[1], vertices[:,0] - center[0])
  idx = np.argsort(angles)
  return vertices[idx]

# Calculates area of ordered vertices of polygon
def polygon_area(vertices):
  total_area = 0
  for idx in range(0,vertices.shape[0]-1):
    total_area += vertices[idx,0]*vertices[idx+1,1] - vertices[idx+1,0]*vertices[idx,1]
  total_area += vertices[-1,0]*vertices[0,1] - vertices[0,0]*vertices[-1,1]
  return 0.5 * abs(total_area)

def isnan(x):
    return x != x

def unique_name():
	return uuid.uuid4().hex[:6]

def make_dir_if_not_exists(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def sum_i_neq_j(x):
	"""Sum over all elements except i-th for all i (used in VIMCO calculation)

	Input:
		x: Variable of size `iw_size` x `batch_size`
	
	Output:
		result: Of size, `iw_size` x `batch_size` (i,j)th element is equal to sum_{k neq i} x_{k,j}
	"""
	iw_size = x.size(0)
	batch_size = x.size(1)

	# TODO: Would torch.expand instead of torch.repeat make this faster?
	inv_mask = (1. - torch.eye(iw_size)
				).unsqueeze(dim=2).repeat(1, 1, batch_size)
	x_masked = torch.mul(x.view(1, iw_size, batch_size), inv_mask)
	return torch.sum(x_masked, dim=1)

def ln_sum_i_neq_j(x):
	"""Sum over all elements except i-th for all i in log-space (used in VIMCO calculation)

	Input:
		x: Variable of size `iw_size` x `batch_size`
	
	Output:
		result: Of size, `iw_size` x `batch_size` (i,j)th element is equal to sum_{k neq i} x_{k,j} in log-space
	"""
	iw_size = x.size(0)
	batch_size = x.size(1)

	# TODO: Would torch.expand instead of torch.repeat make this faster?
	inv_mask = torch.eye(iw_size).unsqueeze(dim=2).repeat(1, 1, batch_size)
	x_masked = x.view(1, iw_size, batch_size) - inv_mask*1000000.0
	return logsumexp(x_masked, dim=1)

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def numify(x):
	return np.round(x.cpu().detach().numpy(), decimals=3)

def numify2(x):
	return x.cpu().detach().numpy()

def stats(v):
  print('min', torch.min(v).cpu().detach().numpy(), 'max', torch.max(v).cpu().detach().numpy(), 'mean', torch.mean(v).cpu().detach().numpy(), 'NaNs', torch.sum(isnan(v)).cpu().detach().numpy(), '-Inf', torch.sum(v==float("-Inf")).cpu().detach().numpy(), '+Inf', torch.sum(v==float("Inf")).cpu().detach().numpy() )

def model_pfd(cell_symb, sample_count, v):
    # update pfd
    pfd = 0
    for key in cell_symb:
        op_cell = cell_symb[key][2] / (sample_count + v)
        lg_p = cell_symb[key][1]
        pfd += op_cell * (10 ** (lg_p))
    # add pfd of empty cell
    pfd += v/(sample_count + v) * 1.0
    print('probability of failure:', pfd)
    return pfd

def model_avg_fail(cell_symb):
    # update avg acc
    avg_acc = 0
    for key in cell_symb:
        lg_p = cell_symb[key][1]
        avg_acc += 10 ** (lg_p)
    avg_acc = avg_acc / len(cell_symb)
    print('average failure:', avg_acc)
    return avg_acc


class record:

    def __init__(self, filename, startTime):

        self.startTime = startTime

        directory = os.path.dirname(filename)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
        self.file = open(filename, "w+")

    def write(self, text):
        self.file.write(text)

    def close(self):
        self.file.close()

    def resetTime(self):
        self.write("reset time at %s\n\n" % (time.time() - self.startTime))
        self.startTime = time.time()


def writeInfo(r, idx, pfd, avg_accuracy, mle_fail):
    r.write("time:%s\n" % (time.time() - r.startTime))
    r.write('--------------------------\n')
    r.write("No. of Samples:%d\n" % (idx))
    r.write("probability of failure: %.5f\n" % (pfd))
    r.write("average failure: %.5f\n" % (avg_accuracy))
    r.write("MLE failure estimation: %.5f\n" % (mle_fail))
    r.write('--------------------------\n')
    r.write('--------------------------\n')

def get_nearest_oppo_dist(X, y, norm, n_jobs=10):
    if len(X.shape) > 2:
        X = X.reshape(len(X), -1)
    p = norm

    def helper(yi):
        return NearestNeighbors(n_neighbors=1,
                                metric='minkowski', p=p, n_jobs=12).fit(X[y != yi])

    nns = Parallel(n_jobs=n_jobs)(delayed(helper)(yi) for yi in np.unique(y))
    ret = np.zeros(len(X))
    for yi in np.unique(y):
        dist, _ = nns[yi].kneighbors(X[y == yi], n_neighbors=1)
        ret[np.where(y == yi)[0]] = dist[:, 0]

    return nns, ret

def plot_label_clusters(latent_data, labels):
    # display a 2D plot of the digit classes in the latent space
    plt.figure(figsize=(12, 10))
    plt.scatter(latent_data[:, 0], latent_data[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

def load_checkpoint(model, model_dir):
    path = os.path.join(model_dir, model.name)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    return epoch