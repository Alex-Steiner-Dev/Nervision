from gensim.models import Word2Vec
import numpy as np
import os

sentences_path = "../Data/Text/sentences.txt"
sentences = []

with open(sentences_path, "r") as f:
    for line in f:
        new_sentence = []
        for i in line.split():
            if not i == "\n":
                new_sentence.append(i)
        sentences.append(new_sentence)

print(sentences)
model = Word2Vec(sentences, min_count=1)

model.save("text_model.bin")