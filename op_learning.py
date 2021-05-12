import torch
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import torch.distributions as dist

class op_learning:
    def __init__(self, y, x_max, x_min, cell_size):
        self.unique_symbs = None
        self.unique_indices = None
        self.unique_counts = None
        self.y = y
        self.unique_y = []
        self.unique_y_pred = None
        self.bins = None
        self.kde = None
        self.x_max = torch.tensor(x_max)
        self.x_min = torch.tensor(x_min)
        self.cell_size = cell_size
        self.cell_interval = torch.tensor((x_max - x_min) / cell_size)

    def init_op(self,x):
        # # use grid search cross-validation to optimize the bandwidth
        # params = {'bandwidth': np.logspace(-1, 1, 20)}
        # grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, n_jobs=10)
        # grid.fit(np.array(x_latent.cpu()))
        #
        # print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
        #
        # # use the best estimator to compute the kernel density estimate
        # kde = grid.best_estimator_
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.26366508987303583).fit(np.array(x.cpu()))
        self.symbs = (x - self.x_min) // self.cell_interval
        self.bins = torch.linspace(self.x_min, self.x_max, self.cell_size + 1)
        self.unique_symbs, self.unique_indices, self.unique_counts = torch.unique(self.symbs.long(), dim=0, return_counts=True, return_inverse=True)


    def updata_op(self,x,y):
        symbs = (x - self.x_min) // self.cell_interval
        symbs = torch.cat((self.unique_symbs, symbs), 0)
        self.unique_symbs, unique_indices, unique_counts = torch.unique(symbs.long(), dim=0, return_counts=True, return_inverse=True)
        print('update op')

    def find_ground_truth(self):
        data_idx = [torch.where((self.symbs == symb).all(axis=1)) for symb in self.unique_symbs]
        y_idx = [self.y[idx] for idx in data_idx]
        aaa = 1
        for idx in y_idx:
            if len(idx) == 0:
                self.unique_y.append('empty')
            elif torch.max(idx) == torch.min(idx):
                self.unique_y.append(idx[0])
            else:
                self.unique_y.append('cross')

    def cal_pred_label(self, loader, cell_points):

        data_inputs = torch.zeros(len(cell_points), 784).cuda()
        self.unique_y_pred = torch.zeros(len(cell_points), dtype=torch.long).cuda()
        data_loader = torch.utils.data.DataLoader(cell_points, batch_size=1000, shuffle=False)
        with torch.no_grad():
            for idx, data in enumerate(data_loader):

                z_projected = loader.g_model.project(data).view(
                    -1, loader.g_model.kernel_num,
                    loader.g_model.feature_size,
                    loader.g_model.feature_size,
                )
                input_points = loader.g_model.decoder(z_projected).data
                input_points = loader.data_resize(input_points, 28)
                input_points = input_points.view(-1, 784)
                pred = torch.argmax(loader.model(loader.data_normalization(input_points)), dim=1)


                self.unique_y_pred[(idx * 1000):((idx + 1) * 1000)] = pred.detach()
                data_inputs[(idx * 1000):((idx + 1) * 1000)] = input_points

        return data_inputs







