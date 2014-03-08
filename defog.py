__author__ = 'Yang'

import numpy as np
from acrm import ApproxCorankingMatrix
from sklearn import datasets, manifold
from scipy.interpolate import griddata


class Defog(object):

    def __init__(self, n_seeds=1000):
        print("init")
        self.seeds = None
        self.seed_values = None
        self.grid_x = None
        self.grid_y = None
        self.grid_z = None
        self.acrm = None
        self.hd = None
        self.ld = None
        self.n_seeds = n_seeds
        self.n_grid = 50j
        np.random.seed(np.random.randint(10))

    def preprocess(self):
        print("preprocess")
        self.hd = self._get_hd_data(100)[0]
        self.ld = self._get_ld_data()
        self.acrm = ApproxCorankingMatrix(k=10)
        self.acrm.preprocess(self.hd, self.ld)

        margin = 0.0
        domain = [
            np.min([np.min(self.ld[:, 0]) - margin, np.max(self.ld[:, 0]) + margin]),
            np.max([np.min(self.ld[:, 1]) - margin, np.max(self.ld[:, 1]) + margin])
        ]
        self.grid_x, self.grid_y = np.mgrid[
            domain[0]:domain[1]:self.n_grid, domain[0]:domain[1]:self.n_grid
        ]
        self.seeds = np.random.uniform(domain[0], domain[1], (self.n_seeds, 2))

    def get_seed_values(self, selected=None, recalculate=True):
        print("getting seed data")
        if selected is None:
            selected = np.argmin(self.acrm.evals)
        if recalculate:
            self.seed_values = [self.acrm.update_data_low(selected, s) for s in self.seeds]
            self.seed_values = self._normalize(np.array(self.seed_values))
        return self.seed_values

    def get_grid_data(self, selected=None, recalculate=True, overview=False, normalize=False):
        print("getting grid data")
        if selected is None:
            selected = np.argmin(self.acrm.evals)
        if recalculate:
            if overview:
                print("raw")
                self.grid_z = griddata(self.ld, self.acrm.evals,      # embeddings as seeds
                                       (self.grid_x, self.grid_y),    # output grid
                                       method='linear')               # options
            else:
                self.get_seed_values(selected)
                self.grid_z = griddata(self.seeds, self.seed_values,  # random seeds
                                       (self.grid_x, self.grid_y),    # output grid
                                       method='linear')                # options
            self._extrapolate()  # extrapolate nans on edge
            if normalize:
                self.grid_z = self._normalize(self.grid_z)
        return self.grid_z

    @staticmethod
    def _normalize(data):
        _min, _max = data.min(), data.max()
        _range = _max - _min
        if _range == 0:
            return data
        else:
            return (data - _min) / _range

    def _get_hd_data(self, size=-1, normalize=True):
        digits = datasets.load_digits(n_class=5)
        if size != -1:  # return sampled
            samples = np.random.randint(digits.data.shape[0], size=size)
            data = self._normalize(digits.data[samples]) if normalize else digits.data[samples]
            return data, digits.target[samples], digits.images[samples]
        else:  # return all
            data = self._normalize(digits.data) if normalize else digits.data
            return data, digits.target, digits.images

    def _get_ld_data(self, normalize=True):
        mds = manifold.MDS(n_components=2, n_init=1, max_iter=100)
        embeddings = mds.fit_transform(self.hd)
        if normalize:
            embeddings = self._normalize(embeddings)
        return embeddings

    def _extrapolate(self):
        # extrapolate the NaNs or masked values in a grid INPLACE using nearest value.
        if np.ma.is_masked(self.grid_z):
            nans = self.grid_z
        else:
            nans = np.isnan(self.grid_z)
        notnans = np.logical_not(nans)
        self.grid_z[nans] = griddata(
            (self.grid_x[notnans], self.grid_y[notnans]), self.grid_z[notnans],
            (self.grid_x[nans], self.grid_y[nans]), method='nearest'
        ).ravel()

if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan, linewidth=400, precision=2, suppress=False)

    defog = Defog()
    defog.preprocess()
    print(defog.get_grid_data())
