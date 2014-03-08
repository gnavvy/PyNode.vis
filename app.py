__author__ = 'ywang'

from flask import Flask, render_template, jsonify
from defog import Defog
import numpy as np

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/getData')
def get_data():
    indices = defog.get_control_point_indices()
    data = defog.get_grid_data()

    return jsonify({'data': {
        'values': data.tolist(),
        'indices': indices.tolist()
    }})

if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan, linewidth=400, precision=2, suppress=False)

    defog = Defog()
    defog.preprocess()
    app.run()
