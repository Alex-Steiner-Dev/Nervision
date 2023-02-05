const express = require('express');
var shell = require('shelljs');
const { Triangle } = require('three');

const app = express();
const PORT = 9000;

app.use(express.static('public'));
app.set('view engine', 'ejs');

app.get('/', function(req, res){
    res.render('index');
});

// Generation
app.get('/generate', function(req, res){
    res.render('generate');
});

async function execute(command){
    await shell.exec(command);
}

app.listen(PORT, function(){
    console.log(`Server is running on port ${PORT}`);
}); 