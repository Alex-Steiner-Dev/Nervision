import gensim
import numpy as np

data = [["a sitting chair"], ["a chair"]]
model = gensim.models.Word2Vec(data, min_count=1)

def word_embedding(text_descriptions):
    output = model.wv[text_descriptions]
    return output