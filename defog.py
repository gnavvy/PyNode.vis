__author__ = 'Yang'

import numpy as np
from numpy import random
from acrm import ApproxCorankingMatrix
from sklearn import datasets, manifold
from scipy.interpolate import griddata


class Defog(object):

    def __init__(self, n_seeds=1000):
        print("init")
        self.seeds = None
        self.values = None
        self.grid_x = None
        self.grid_y = None
        self.acrm = None
        self.hd = None
        self.ld = None
        self.n_seeds = n_seeds
        self.n_grid = 50j
        np.random.seed(np.random.randint(10))

    def preprocess(self):
        print("preprocess")
        self.hd = self._get_hd_data()[0]
        self.ld = self._get_ld_data()
        self.acrm = ApproxCorankingMatrix(k=6)
        self.acrm.preprocess(self.hd, self.ld)

        margin = 0.1
        domain = [
            np.min([np.min(self.ld[:, 0]) - margin, np.max(self.ld[:, 0]) + margin]),
            np.max([np.min(self.ld[:, 1]) - margin, np.max(self.ld[:, 1]) + margin])
        ]
        self.grid_x, self.grid_y = np.mgrid[
            domain[0]:domain[1]:self.n_grid, domain[0]:domain[1]:self.n_grid
        ]
        self.seeds = np.random.uniform(domain[0], domain[1], (self.n_seeds, 2))

    def get_data(self, selected=-1):
        print("getting data")
        if selected == -1:
            selected = np.argmin(self.acrm.evals)
        self.values = [self.acrm.update_data_low(selected, s) for s in self.seeds]
        self.values = self._normalize(np.array(self.values))
        return griddata(self.seeds, self.values, (self.grid_x, self.grid_y),
                        fill_value=0, method='cubic')

    @staticmethod
    def _normalize(data):
        _min = data.min()
        _range = data.max() - _min
        return data if _range == 0 else (data - _min) / _range

    def _get_hd_data(self, size=-1, normalize=True):
        digits = datasets.load_digits(n_class=5)
        if size != -1:
            samples = random.randint(digits.data.shape[0], size=size)
            data = self._normalize(digits.data[samples]) if normalize else digits.data[samples]
            return data, digits.target[samples], digits.images[samples]
        else:
            data = self._normalize(digits.data) if normalize else digits.data
            return data, digits.target, digits.images

    def _get_ld_data(self, normalize=True):
        mds = manifold.MDS(n_components=2, n_init=1, max_iter=100)
        embeddings = mds.fit_transform(self.hd)
        if normalize:
            embeddings = self._normalize(embeddings)
        return embeddings
