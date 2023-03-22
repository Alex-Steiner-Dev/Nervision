from flask import Flask, render_template, request
#from predict import generate
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

@app.route('/generation', methods=['GET', 'POST'])
def generation():
    if request.method == "POST":
        pass
        #generate(request.form.get("input-bar"))

    return render_template('generation.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)