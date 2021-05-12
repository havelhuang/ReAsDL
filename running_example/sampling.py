import numpy as np
from scipy.stats import norm,sem
from math import sqrt
import multiprocessing
from joblib import delayed, Parallel

def density_eval(kde,x):
    return np.exp(kde.score_samples(x))

def kde_variance(kdes, x):
    NUM_CPUS = int(0.875 * multiprocessing.cpu_count())
    print('threads num: ',NUM_CPUS)
    op = Parallel(n_jobs=NUM_CPUS)(delayed(density_eval)(kde,x) for kde in kdes)
    return np.var(op, ddof=1, axis=0)

def failure(y_pred,y_true,fail_type):
    if fail_type == 'a':
        return y_pred != y_true
    elif fail_type == 'p':
        return  (y_pred != y_true) and (y_true == 1)
    else:
        return  (y_pred != y_true) and (y_true == 0)

# Monte Carlo sampling to verify cell astuteness
def monte_carlo(position, y, cell_size, model, fail_type):
    # fail type: false positive 'p'
    #            false negative 'n'
    #            all miss-classification 'a'
    np.random.seed(100)
    k = 10000
    if y == 'cross':
        print('lambda is 1')
        return [1,0]
    else:
        x_low = position - cell_size/2
        x_high = position + cell_size/2
        x_samples = np.random.uniform(x_low, x_high, (k,2))
        y_predict = model.predict(x_samples)
        if y == 'empty':
            y = np.bincount(y_predict).argmax()
            y_samples = y * np.ones(len(x_samples))
        else:
            y_samples = y * np.ones(len(x_samples))

        y_fail = [1 if failure(a,b,fail_type) else 0 for a, b in zip(y_predict, y_samples)]
        y_fail = np.array(y_fail)
        mean, var = np.mean(y_fail), np.var(y_fail, ddof=1)
        print('mean:%f, var:%f' %(mean, var))
        return [mean, var/len(y_fail)]

