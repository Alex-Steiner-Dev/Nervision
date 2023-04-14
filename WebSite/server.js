const express = require('express');

const mongoose = require("mongoose");
const bodyParser = require("body-parser");

const cookieParser = require('cookie-parser');
const session = require('express-session');

const app = express();

app.use(bodyParser.urlencoded({extended: true})); 

app.use(cookieParser());
app.use(session({secret: "ghfdjkgdhkqEWQE9Qjkl329490ASH<A!àfsdfkJjgxDIHk284329FHDS_DASDJW_Dsdfdmsa"}));

app.set('view engine', 'ejs'); 
app.use(express.static(__dirname + '/static'));

const userSchema ={
    userName: String,
    psw: String
}
  
const users = mongoose.model("users", userSchema)

app.get('/',async function(req, res){
    if(req.session.username != null){  
        users.find({ userName: req.session.username, psw:req.session.psw}).then( function(x){
            res.render('index-logged');
        });
    }

    else{
        res.render('index');
    }
});

app.get('/login', function(req, res){
    res.render('login');
});

app.post('/login', async function(req, res){
    const username = req.body.username;
    const psw = req.body.psw;

    users.find({ userName: username, psw:psw}).then(function(x){
        if(x.length != 0){
            req.session.username = username;
            req.session.psw = psw;
            req.session.user = true;

            res.render('index-logged');
        }
        else{
            res.send("Wrong data");
        }
    });

});

app.get('/signup', function(req, res){
    res.render('signup');
});

app.post('/signup', function(req, res){
    const name = req.body.username;
    const psw = req.body.psw;

    const newUser = new users({
        userName: name,
        psw: psw
    })

    users.create(newUser);

    res.render('login');
});

app.get('/download', function(req, res){
    var path = "static/generation.glb";
});

app.listen(port=8080, function(){
    console.log("Server is running on port 8080");

    mongoose.connect("mongodb+srv://francesco:x3UWKnnEYqVrcQNy@cluster.mpqus65.mongodb.net/")
    console.log("Connect to the DB!");
});