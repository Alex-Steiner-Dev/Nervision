from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/generate', methods = ['POST'])
def generate():
    if request.method == 'POST':
        output = request.form.get('prompt', '')
        return ""
    else:
        return "You are visiting this website in the wrong way"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1000)