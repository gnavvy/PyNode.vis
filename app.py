__author__ = 'ywang'

from flask import Flask, render_template, jsonify
import numpy as np
import json

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/getData')
def get_data():
    return jsonify({'values': np.random.random_sample((50, 50)).tolist()})


if __name__ == '__main__':
    app.run()
