from joblib import Parallel, delayed
import numpy as np
from sklearn.neighbors import NearestNeighbors
from Mnist import Mnist
from Cifar10 import Cifar10

CUDA = False

def get_nearest_oppo_dist(X, y, tstX, tsty, norm, n_jobs=10):
    if len(X.shape) > 2:
        X = X.reshape(len(X), -1)
        tstX = tstX.reshape(len(tstX), -1)
    p = norm

    def helper(yi):
        return NearestNeighbors(n_neighbors=1,
                                metric='minkowski', p=p, n_jobs=12).fit(X[y != yi])

    nns = Parallel(n_jobs=n_jobs)(delayed(helper)(yi) for yi in np.unique(y))
    ret = np.zeros(len(X))
    tst_ret = np.zeros(len(tstX))
    for yi in np.unique(y):
        dist, _ = nns[yi].kneighbors(X[y == yi], n_neighbors=1)
        ret[np.where(y == yi)[0]] = dist[:, 0]

        dist, _ = nns[yi].kneighbors(tstX[tsty == yi], n_neighbors=1)
        tst_ret[np.where(tsty == yi)[0]] = dist[:, 0]

    return nns, ret, tst_ret


loader = Mnist(CUDA)

trnX = np.array(loader.x_train)
trnX = np.resize(trnX, (trnX.shape[0], 28, 28, 1))
trny = np.array(loader.y_train)

tstX = np.array(loader.x_test)
tstX = np.resize(tstX, (tstX.shape[0], 28, 28, 1))
tsty = np.array(loader.y_test)

nns_linf, mnist_dists_linf, tst_mnist_dists_linf = get_nearest_oppo_dist(trnX, trny, tstX, tsty, np.inf, n_jobs=5)
print(mnist_dists_linf.min(), mnist_dists_linf.mean())
print(tst_mnist_dists_linf.min(), tst_mnist_dists_linf.mean())