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
        self.data_high = None
        self.data_low = None
        self.rank_matrix_high = None
        self.rank_matrix_low = None

        self.engine = None
        self.coranking_matrices = None
        self.evals = None

    def preprocess(self, hd, ld):
        self.data_high = hd
        self.data_low = ld
        self.n, self.m = self.data_high.shape

        self.engine = Engine(self.m, lshashes=[
            hashes.RandomBinaryProjections('rbp', self.qdim)
            # hashes.RandomDiscretizedProjections('rdp', self.qdim, 5)
        ], vector_filters=[
            filters.NearestFilter(self.k + 1)
        ])
        self.coranking_matrices = np.array([np.zeros((self.k, self.k))] * self.n)
        self.evals = np.zeros(self.n)

        self.rank_matrix_low = self._get_exact_rank_matrix(self.data_low)
        self.rank_matrix_high = self._get_exact_rank_matrix(self.data_high)
        # self.rank_matrix_high = self._get_approx_rank_matrix(self.data_high)
        self._update_eval()

    def evaluate(self, idx=-1):  # return overall quality by default
        if idx == -1:
            return np.mean(self.evals)
            # return np.sum(self.evals) / self.n
        elif 0 <= idx < self.n:
            return self.evals[idx]
        else:
            print("evaluation index out of boundary")
            exit(1)

    def update_data_low(self, idx, entry):
        self.data_low[idx] = entry
        ranks_new = self._get_exact_rank_matrix(self.data_low)
        diff_idx = np.unique(np.where(self.rank_matrix_low != ranks_new)[0])
        self.rank_matrix_low = ranks_new
        self._update_eval(diff_idx)
        print()
        return self.evaluate()

    def _get_approx_rank_matrix(self, data):
        # project and hash each data point into the hash table
        [self.engine.store_vector(data[i], i) for i in range(self.n)]
        rank_matrix = np.array([self._get_ann(entry) for entry in data])
        return rank_matrix[:, 1:]  # remove first column -> self index

    def _get_exact_rank_matrix(self, data):
        # todo change to use kd-tree instead of ball tree for performance comparison
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(data)
        rank_matrix = nn.kneighbors(data, return_distance=False)
        return rank_matrix[:, 1:]  # remove first column -> self index

    def _get_ann(self, entry):
        ann = np.array([nn[1] for nn in self.engine.neighbours(entry)])
        if ann.shape[0] < self.k + 1:
            ann.resize(self.k + 1, refcheck=False)
        return ann

    # todo bottleneck
    def _update_coranking_matrices(self, indices=None):
        if indices is None:  # reset non-zero values
            self.coranking_matrices[self.coranking_matrices != 0.] = 0.
            indices = range(self.n)

        [self._update_matrix(idx) for idx in indices]

    def _update_matrix(self, idx):
        self.coranking_matrices[idx][self.coranking_matrices[idx] != 0.] = 0.

        for rank in range(self.k):
            p = np.where(self.rank_matrix_high[idx] == self.rank_matrix_low[idx][rank])[0]
            q = np.where(self.rank_matrix_low[idx] == self.rank_matrix_high[idx][rank])[0]
            if p.shape[0] != 0:
                self.coranking_matrices[idx, p[0], rank] = 1
            if q.shape[0] != 0:
                self.coranking_matrices[idx, rank, q[0]] = 1

        self.evals[idx] = np.sum(self.coranking_matrices[idx]) / self.k

    def _update_eval(self, indices=None):
        if indices is None:
            self.evals[self.evals != 0] = 0
            indices = range(self.n)

        for idx in indices:
            high = self.rank_matrix_high[idx]
            low = self.rank_matrix_low[idx]
            n_common = np.intersect1d(high, low).shape[0]
            self.evals[idx] = n_common / self.k


