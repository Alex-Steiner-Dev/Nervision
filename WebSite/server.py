from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/generation')
def generation():
    return render_template('generation.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)