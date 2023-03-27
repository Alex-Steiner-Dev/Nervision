from flask import Flask, render_template, request, send_file
from predict import generate
import pymongo

import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

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

        return "Ok"

    else:
        return render_template('signup.html')

@app.route('/generation', methods=['GET', 'POST'])
def generation():
    if request.method == "POST":
        generate(request.form.get("input-bar"))
        return render_template('generated.html')

    return render_template('generation.html')

@app.route('/download')
def download ():
    path = "static/generation.obj"
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)