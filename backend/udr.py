__author__ = 'Yang'

import time
import numpy as np
from numpy import random
from acrm import ApproxCorankingMatrix
from sklearn import datasets, manifold
from scipy.interpolate import griddata

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput


def _normalize(data):
    _min = data.min()
    _range = data.max() - _min
    return data if _range == 0 else (data - _min) / _range


def get_hd_data(size=-1, normalize=True):
    digits = datasets.load_digits(n_class=5)
    if size != -1:
        samples = random.randint(digits.data.shape[0], size=size)
        data = _normalize(digits.data[samples]) if normalize else digits.data[samples]
        return data, digits.target[samples], digits.images[samples]
    else:
        data = _normalize(digits.data) if normalize else digits.data
        return data, digits.target, digits.images


def get_ld_data(normalize=True):
    embeddings = manifold.MDS(n_components=2, n_init=1, max_iter=100).fit_transform(data_high)
    if normalize:  # change to use _normalize
        data_min, data_max = embeddings.min(), embeddings.max()
        embeddings = (embeddings - data_min) / (data_max - data_min)
    return embeddings


if __name__ == "__main__":
    with PyCallGraph(output=GraphvizOutput(output_file=time.strftime("%H%M%S")+".png")):
        np.set_printoptions(threshold=np.nan, linewidth=400)

        print("loading high-dimensional data")
        data_high, labels, images = get_hd_data(64)
        n_samples, n_dim = data_high.shape
        print("n = %d, m = %d" % (n_samples, n_dim))

        print("embedding low-dimensional data")
        data_low = get_ld_data()

        print("evaluating")
        acrm = ApproxCorankingMatrix(k=6, qdim=5)
        acrm.preprocess(data_high, data_low)
        evals = acrm.evals

        print("building coordinates")
        margin = 0.1
        x_min, x_max = np.min(data_low[:, 0]) - margin, np.max(data_low[:, 0]) + margin
        y_min, y_max = np.min(data_low[:, 1]) - margin, np.max(data_low[:, 1]) + margin
        domain = [np.min([x_min, y_min]), np.max([x_max, y_max])]

        n_seed = 1000
        np.random.seed(3)
        seeds = np.random.uniform(x_min, x_max, (n_seed, 2))

        print("interpolating")
        res = 50j  # grid axis dimension / resolution
        grid_x, grid_y = np.mgrid[domain[0]:domain[1]:res, domain[0]:domain[1]:res]

        selected = np.argmin(evals)  # select the point w/ lowest eval

        seed_z = np.array([acrm.update_data_low(selected, seed) for seed in seeds])

        _normalize(seed_z)

        values = griddata(seeds, seed_z, (grid_x, grid_y), fill_value=0, method='cubic')




