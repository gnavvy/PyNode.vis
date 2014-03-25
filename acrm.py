__author__ = 'ywang'

import numpy as np
from sklearn.neighbors import NearestNeighbors
from nearpy import Engine, hashes, filters


class ApproxCorankingMatrix:

    def __init__(self, k=5, qdim=10):
        self.n = 0  # number of data points
        self.m = 0  # dimension of the vector description of a data point
        self.k = k  # number of nearest neighbors
        self.qdim = qdim  # dimension of the quantize vector
        self.data_hd = None
        self.data_ld = None
        self.ranks_hd = None
        self.ranks_ld = None
        self.engine = None
        self.quality = None

    def preprocess(self, hd, ld):
        self.data_hd = hd
        self.data_ld = ld
        self.n, self.m = self.data_hd.shape
        self.ranks_ld = self._get_exact_ranks(self.data_ld)
        self.ranks_hd = self._get_exact_ranks(self.data_hd)
        self.engine = Engine(self.m, lshashes=[
            hashes.RandomBinaryProjections('rbp', self.qdim)
        ], vector_filters=[
            filters.NearestFilter(self.k + 1)
        ])
        self.quality = np.zeros(self.n)
        self._calculate_quality()

    def _get_overall_quality(self):
        return np.mean(self.quality)

    def update_data_low(self, idx, entry):
        self.data_ld[idx] = entry
        ranks_new = self._get_exact_ranks(self.data_ld)
        diff_idx = np.unique(np.where(self.ranks_ld != ranks_new)[0])
        self.ranks_ld = ranks_new
        self._calculate_quality(diff_idx)
        return self._get_overall_quality()

    def _get_approx_ranks(self, data):
        # project and hash each data point into the hash table
        [self.engine.store_vector(data[i], i) for i in range(self.n)]
        rank_matrix = np.array([self._get_ann(entry) for entry in data])
        return rank_matrix[:, 1:]  # remove first column -> self index

    def _get_exact_ranks(self, data):
        # todo change to use kd-tree instead of ball tree for performance comparison
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(data)
        rank_matrix = nn.kneighbors(data, return_distance=False)
        return rank_matrix[:, 1:]  # remove first column -> self index

    def _get_ann(self, entry):
        ann = np.array([nn[1] for nn in self.engine.neighbours(entry)])
        if ann.shape[0] < self.k + 1:
            ann.resize(self.k + 1, refcheck=False)
        return ann

    def _calculate_quality(self, indices=None):
        if indices is None:
            indices = range(self.n)

        for idx in indices:
            high = self.ranks_hd[idx]
            low = self.ranks_ld[idx]
            n_common = np.intersect1d(high, low).shape[0]
            self.quality[idx] = n_common * 1.0 / self.k
