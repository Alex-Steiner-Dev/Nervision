from flask import Flask
from flask import request

from getImage import *

app = Flask(__name__)

@app.route('/generate', methods = ['POST'])
def index():
    if request.method == 'POST':
        output = request.form.get('prompt', '')
        return getImage(output)
    else:
        return "You are visiting this website in the wrong way"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1000)