import json
import en_core_web_sm
from text_to_vec import *

nlp = en_core_web_sm.load()

f = open("captions.json")
data = json.load(f)
text_embeddings = []
ids = []

for i, itObject in enumerate(data):
        if itObject['desc'].split('.')[0].find(".") != -1:
            text_embeddings.append(nlp((itObject['desc'])))
        else:
            text_embeddings.append(nlp((itObject['desc'].split('.')[0])))
        ids.append(itObject['mid'])

f.close()
        
text = "cocktail table"

def return_id(text):
    text = nlp(text)

    x = 0
    y = ""

    for i, j in enumerate(text_embeddings):
         if text.similarity(j)> x:
              x = nlp(j)
              y = ids[i]

    return y

print(return_id(text))