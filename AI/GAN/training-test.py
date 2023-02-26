from VAE import *
from word_embedding import *

#autoencoder = VAE(box_size=32).build_vae()
word_embedding = WordEmbedding()
word_embedding.generate_model()
print(word_embedding.generate_word_embedding("a modern chair"))