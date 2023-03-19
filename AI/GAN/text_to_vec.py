from gensim.models import Word2Vec

def text_to_vec(sentence):
    sentences = [sentence.split()]
    model = Word2Vec(sentences, vector_size=128, window=5, min_count=1, workers=2)
    vector = model.wv[sentence.split()[0]]
    
    return vector

text_to_vec("An airplane with wings")