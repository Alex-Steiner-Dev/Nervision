from flask import Flask
from flask import render_template
from flask import request

import sys

sys.path.insert(1, '../AI/')
sys.path.insert(2, '../AI/Train')
sys.path.insert(3, '../AI/Generate Model')

from evaluation import *
from text_process import *
from model_generation import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'GET':
        return render_template('generate.html')
    elif request.method == 'POST':
        generate_model(process_text(request.form.get('prompt')))
        return f"Generating {process_text(request.form.get('prompt'))}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9000)