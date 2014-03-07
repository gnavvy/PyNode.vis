__author__ = 'ywang'

from flask import Flask, render_template, jsonify
from defog import Defog

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/getData')
def get_data():
    return jsonify({'values': defog.get_data().tolist()})

if __name__ == '__main__':
    defog = Defog()
    defog.preprocess()
    app.run()
