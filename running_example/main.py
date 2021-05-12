import pandas as pd
import numpy as np
import timeit
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from sampling import monte_carlo,kde_variance
from sklearn.utils import resample
from math import sqrt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import multiprocessing

# parallel prediction of samples' density score
def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))

# calculate r-separation distance of dataset
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

# display a 2D plot of the digit classes in the latent space
def plot_label_clusters(latent_data, labels):
    plt.figure(figsize=(12, 10))
    plt.scatter(latent_data[:, 0], latent_data[:, 1], c=labels)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    plt.xlabel("x[0]",fontsize=15)
    plt.ylabel("x[1]",fontsize=15)
    plt.title('C',fontsize=30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig("2.png",bbox_inches='tight')
    plt.show()

# display density in 2d space
def plot_2d_density(kde):
    # Create meshgrid
    x1, x2 = np.mgrid[0:1:100j, 0:1:100j]
    positions = np.vstack([x1.ravel(), x2.ravel()])
    f = np.reshape(np.exp(kde.score_samples(positions.T)), x1.shape)
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection='3d')
    w = ax.plot_wireframe(x1, x2, f)
    ax.set_xlabel('x1',fontsize=15)
    ax.set_ylabel('x2',fontsize=15)

    ax.set_zlabel('PDF',fontsize=15)
    # ax.set_title('C',fontsize=30)
    plt.savefig("3.png", bbox_inches='tight')
    plt.show()


def main():
    # load training data from file
    excel_file = 'trainingdata_a.xls'
    df = pd.read_excel(excel_file)
    x = df[['x_i1','x_i2']].to_numpy()
    y = df['l_i'].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    clf = RandomForestClassifier(n_estimators=10, random_state=5)
    clf.fit(x_train, y_train)
    print("Training set score: %f" % clf.score(x_train, y_train))
    print("Test set score: %f" % clf.score(x_test, y_test))

    # define cell size for each dimension
    cell_amount = 500
    cell_size = 1/cell_amount
    symbs = (x-0)//cell_size
    # kde to fit distribution of training data
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(x)
    ###############################################
    # plot_2d_density(kde)
    # nns, ret = get_nearest_oppo_dist(x, y, np.inf, n_jobs=10)
    # print(ret.min(), ret.mean())
    # plot_label_clusters(x, y)
    ###############################################
    # use first k cells to estimate reliability
    k = 300000
    # use cell central point to estimate cell operational profile
    x_s = 0 + cell_size/2
    x_e = 1 - cell_size/2
    x1, x2 = np.mgrid[x_s:x_e:complex(0,cell_amount), x_s:x_e:complex(0,cell_amount)]
    positions = np.vstack([x1.ravel(), x2.ravel()]).T
    cell_op = np.exp(parrallel_score_samples(kde, positions))
    order = cell_op.argsort()
    order = order[::-1]
    positions = positions[order[:k]]
    cell_op = cell_op[order[:k]]
    scale_op = sum(cell_op)
    cell_op = cell_op / scale_op
    # find the bootstrap variance of kde
    B = 100
    kdes_b = []
    for i in range(B):
        x_b = resample(x, n_samples=int(0.5*len(x)), random_state=i)
        kde_b = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(x_b)
        kdes_b.append(kde_b)
    cell_op_var = kde_variance(kdes_b,positions) / (scale_op*scale_op)

    # find the ground truth label of first k cells
    y_cell = []
    positions_symbs = (positions -0)//cell_size
    data_idx = [np.where((symbs == symb).all(axis=1)) for symb in positions_symbs]
    y_idx = [y[idx] for idx in data_idx]
    for idx in y_idx:
        if len(idx) == 0:
            y_cell.append('empty')
        elif np.max(idx) == np.min(idx):
            y_cell.append(idx[0])
        else:
            y_cell.append('cross')

    # verify the robustness of each cell
    lambda_cell = [monte_carlo(p, y, cell_size, clf, 'p') for p,y in zip(positions,y_cell)]
    lambda_cell_var = np.array(lambda_cell)[:,1]
    lambda_cell = np.array(lambda_cell)[:, 0]
    reliability = sum(cell_op * lambda_cell)
    reliability_var = sum(cell_op * cell_op * lambda_cell_var + lambda_cell * lambda_cell * cell_op_var + lambda_cell_var * cell_op_var)
    print("Model reliability mean: %f" % reliability)
    print('Model reliability std: %f' % sqrt(reliability_var))

    # compute the confidence interval
    confidence = 0.975
    z = norm.ppf(confidence)
    print("Model reliability %f confidence interval: [0, %f]" % (confidence, reliability+z*sqrt(reliability_var)))

if __name__ == '__main__':
    main()


