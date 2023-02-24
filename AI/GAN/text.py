import gensim
import numpy as np

def word_embedding(text_descriptions):
    data = [["the", "chair"], ["a", "chair"]]
    model = gensim.models.Word2Vec(data, min_count=5)
    text_descriptions = [text.split() for text in text_descriptions]
    output = model.wv[text_descriptions]
    return output[0]