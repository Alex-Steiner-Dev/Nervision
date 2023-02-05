from flask import Flask
from flask import render_template
from flask import request
import sys

sys.path.insert(1, '../AI/train')
sys.path.insert(2, '../AI/')

from evaluation import *
from get_prompt import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'GET':
        return render_template('generate.html')
    elif request.method == 'POST':
        return (f"Generating: {correct_prompt(request.form.get('prompt'))}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9000)