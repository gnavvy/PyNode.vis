__author__ = 'ywang'

from flask import Flask, render_template, jsonify
import numpy as np

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/getData')
def get_data():
    mean = [
        2 + np.random.uniform(-1, 1) * .2,
        2 + np.random.uniform(-1, 1) * .2
    ]
    cov = [
        [.2 + np.random.uniform(-1, 1) * .05, 0],
        [0, .2 + np.random.uniform(-1, 1) * .05]
    ]

    n_sample = 5000

    x, y = np.random.multivariate_normal(mean, cov, n_sample).T

    x_range = [0.0, 4.0]
    y_range = [0.0, 4.0]
    grid_dim = [50, 50]
    x_tick = (x_range[1] - x_range[0]) / grid_dim[0]
    y_tick = (y_range[1] - y_range[0]) / grid_dim[1]

    data = np.zeros(grid_dim[0]*grid_dim[1]).reshape((grid_dim[0], grid_dim[1]))
    for i in range(n_sample):
        xi = np.minimum(np.floor(x[i] / x_tick + 0.5).astype(int), 49)
        yi = np.minimum(np.floor(y[i] / y_tick + 0.5).astype(int), 49)
        if 0 <= xi < 50 and 0 <= yi < 50:
            data[yi][xi] += 1

    data /= np.max(data)
    return jsonify({'values': data.tolist()})


if __name__ == '__main__':
    # get_data()
    app.run()
