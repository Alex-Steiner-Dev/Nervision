from gensim.models import Word2Vec
import numpy as np
import os

class WordEmbedding:
    def __init__(self):
        self.sentences_path = "../Data/Text/wiki.txt" 
        self.model_path = "../Trained Models/Text/text_model.bin"

    def generate_model(self):
        sentences = []

        with open(self.sentences_path, "r") as f:
            for line in f:
                new_sentence = []
                for i in line.split():
                    if not i == "\n":
                        new_sentence.append(i)
                sentences.append(new_sentence)

        model = Word2Vec(sentences, min_count=1)

        model.save(self.model_path)

    def generate_word_embedding(self, text):
        try:
            model = Word2Vec.load(self.model_path)

            words = text.split()
            word_embeddings = [model.wv[w] for w in words]

            return word_embeddings
        
        except:
            return "I am sorry, your requested model wasn't trained on our dataset!"