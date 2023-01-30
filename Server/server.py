from flask import Flask
from getImage import *

app = Flask(__name__)

@app.route('/')
def index():
    return getImage("desert with cacti")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')