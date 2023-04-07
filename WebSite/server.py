from flask import Flask, render_template, request, send_file, session
from flask_session import Session
from predict import generate
import pymongo

import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

client = pymongo.MongoClient(os.getenv("MONGO_URI"))
db = client.db

print("Connected to the DB!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        mail = request.form.get("mail")
        psw = request.form.get("psw")

        user_found = db.users.find_one({ "mail" : mail, "psw": psw})

        try:
            if len(user_found) > 0:
                session["mail"] = mail
                session["psw"] = psw

                return render_template('index.html')
        except:
            return "Wrong data!"
    else:
        return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == "POST":
        mail = request.form.get("mail")
        psw = request.form.get("psw")

        db.users.insert_one({ "mail" : mail, "psw": psw})

        return render_template('login.html')

    else:
        return render_template('signup.html')

@app.route('/logout')
def logout():
    session["mail"] = None
    session["psw"] = None

    return render_template('index.html')    

@app.route('/generation', methods=['GET', 'POST'])
def generation():
    if request.method == "POST":
        generate(request.form.get("input-bar"))
        return render_template('generated.html')

    return render_template('generation.html')

@app.route('/download')
def download ():
    path = "static/generation.glb"
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)