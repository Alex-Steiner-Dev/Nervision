from flask import Flask
from flask import request

from getImage import *

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def index():
    if request.method == 'POST':
        x = request.form.get('prompt', '')
        print(x)
        return getImage(x)
    else:
        return "ciao"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1000)