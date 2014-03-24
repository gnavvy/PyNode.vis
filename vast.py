__author__ = 'Yang'

import numpy as np
from acrm import ApproxCorankingMatrix
from sklearn import (manifold, datasets, decomposition, ensemble, lda, random_projection)
from scipy.interpolate import griddata

from matplotlib import pyplot


class Defog(object):

    def __init__(self, n_seeds=1000):
        print("init")
        self.seeds = None
        self.seed_values = None
        self.domain = None
        self.grid_x = None
        self.grid_y = None
        self.grid_z = None
        self.acrm = None
        self.hd = None
        self.ld = None
        self.n_seeds = n_seeds
        self.selected = None

        # temp
        self.labels = None
        self.images = None

        np.random.seed(np.random.randint(10))

    def preprocess(self):
        # load data
        self.hd, self.labels, self.images = self._get_hd_data(n_data=64)
        self.ld = self._get_ld_data(init_condition=self.ld)

        # calcualte quality
        self.acrm = ApproxCorankingMatrix(k=10)
        self.acrm.preprocess(self.hd, self.ld)

        # generate seed for interpolation
        margin = 0.1
        self.domain = [
            np.min([np.min(self.ld[:, 0] - margin), np.max(self.ld[:, 0]) + margin]),
            np.max([np.min(self.ld[:, 1] - margin), np.max(self.ld[:, 1]) + margin])
        ]
        self.seeds = np.random.uniform(self.domain[0], self.domain[1], (self.n_seeds, 2))

        # seed corner points for complete domain
        self.seeds[0][0], self.seeds[0][1] = self.domain[0], self.domain[0]
        self.seeds[1][0], self.seeds[1][1] = self.domain[0], self.domain[1]
        self.seeds[2][0], self.seeds[2][1] = self.domain[1], self.domain[0]
        self.seeds[3][0], self.seeds[3][1] = self.domain[1], self.domain[1]

    def get_seed_values(self, recalculate=True):
        if self.selected is None:
            self.selected = np.argmin(self.acrm.evals)

        if recalculate:
            selected_backup = self.ld[self.selected]  # backup current pos for selected point
            self.seed_values = [self.acrm.update_data_low(self.selected, s) for s in self.seeds]
            self.seed_values = self._normalize(np.array(self.seed_values))
            self.acrm.update_data_low(self.selected, selected_backup)  # restore backup

        return self.seed_values

    @staticmethod
    def _normalize(data):
        _min, _max = data.min(), data.max()
        _range = _max - _min
        if _range == 0:
            return data
        else:
            return (data - _min) / _range

    def _get_hd_data(self, n_data=-1, n_class=4, normalize=True):
        digits = datasets.load_digits(n_class=n_class)

        if n_data != -1:  # return sampled
            if not n_data % n_class == 0:
                print("please specify a n_data divisible by n_class.")
                exit(1)

            n_samples_per_class = n_data // n_class

            samples = []
            for d in range(n_class):
                count = 0
                for idx in range(digits.target.shape[0]):
                    if digits.target[idx] == d and count < n_samples_per_class:
                        samples.append(idx)
                        count += 1

            data = self._normalize(digits.data[samples]) if normalize else digits.data[samples]
            return data, digits.target[samples], digits.images[samples]
        else:  # return all
            data = self._normalize(digits.data) if normalize else digits.data
            return data, digits.target, digits.images

    def _get_ld_data(self, normalize=True, init_condition=None):
        model = manifold.MDS(n_components=2, max_iter=100)
        # model = manifold.Isomap(n_neighbors=25, n_components=2)
        # model = manifold.SpectralEmbedding(n_neighbors=15, n_components=2)
        # model = manifold.LocallyLinearEmbedding(n_neighbors=15, n_components=2, method='hessian')
        # model = manifold.SpectralEmbedding(n_components=2, random_state=1, eigen_solver="arpack")
        # model = lda.LDA(n_components=2)
        # model = decomposition.RandomizedPCA(n_components=2)

        embedding = model.fit_transform(self.hd, init=init_condition)
        if normalize:
            embedding = self._normalize(embedding)
        return embedding


class Vast():

    def __init__(self):
        self.fig, self.ax = pyplot.subplots(figsize=(9, 8))
        self.fig.canvas.mpl_connect('pick_event', self.onpick)

        self.defog = Defog()
        self.defog.preprocess()

    def onpick(self, event):
        self.defog.selected = event.ind[0]
        self.redraw()

    def redraw(self, recalc=True):
        self.ax.clear()
        ld = self.defog.ld.copy()  # defog.ld will be changed

        X = self.defog.seeds[:, 0]
        Y = self.defog.seeds[:, 1]
        Z = self.defog.get_seed_values(recalculate=recalc)

        self.defog.ld = ld  # restore defog.ld

        min_, max_ = Z.min(), Z.max()
        range_ = max_ - min_
        scale_ = 0.25

        norm_fg = pyplot.Normalize(vmin=0.0, vmax=max_ * 0.8)
        norm_bg = pyplot.Normalize(vmin=min_ - scale_ * range_, vmax=max_ - scale_ * range_)

        cmap_fg = pyplot.cm.get_cmap('gray')
        cmap_bg = pyplot.cm.get_cmap('GnBu_r')

        self.ax.set_xlim(self.defog.domain)
        self.ax.set_ylim(self.defog.domain)

        contour = self.ax.tricontour(X, Y, Z, 10, linewidths=0.2, colors='k')
        heatmap = self.ax.tricontourf(X, Y, Z, 100, cmap=cmap_bg, norm=norm_bg)
        # self.fig.colorbar(heatmap, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        x, y = ld[self.defog.selected]
        target = self.ax.scatter(x, y, c='r', marker='o', alpha=0.8, s=600, linewidths=0)
        points = self.ax.scatter(ld[:, 0], ld[:, 1], c=self.defog.acrm.evals, s=300,
                                 marker='o', cmap=cmap_fg, norm=norm_fg, picker=True)

        for i in range(ld.shape[0]):
            c = 'k' if self.defog.acrm.evals[i] > 0.3 else 'w'
            self.ax.text(ld[i, 0] - .01, ld[i, 1] - .01, str(self.defog.labels[i]), color=c)

        self.fig.canvas.draw()

        # filename = "%i.png" % round
        # pyplot.savefig(filename)


if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan, linewidth=400, precision=2, suppress=False)

    vast = Vast()
    vast.redraw()
    pyplot.show()
