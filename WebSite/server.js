const express = require('express');

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

app.listen(PORT, function(){
    console.log(`Server is running on port ${PORT}`);
}); 