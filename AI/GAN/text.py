import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

word_embedding_model = tf.keras.models.load_model('word_embedding_model.h5')

def word_embedding(text_descriptions):
    # Create a tokenizer to convert text to sequences of integer indices
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_descriptions)
    sequences = tokenizer.texts_to_sequences(text_descriptions)

    # Pad the sequences to a fixed length
    max_seq_len = 10
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='post')

    # Load a pre-trained word embedding model
    embedding_dim = 100

    word_embeddings = word_embedding_model.predict(padded_sequences)

    return word_embedding