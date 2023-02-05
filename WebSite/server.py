from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    return render_template('generate.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9000)