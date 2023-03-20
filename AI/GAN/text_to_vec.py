from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk
from textblob import TextBlob

#nltk.download('stopwords')
#nltk.download('punkt')

def process_text(prompt):
    prompt = prompt.lower()
    prompt = correct_prompt(prompt)
    prompt = remove_unicode(prompt)
    prompt = remove_stop_words(prompt)
    return prompt

def remove_stop_words(prompt):
    word_tokens = word_tokenize(prompt)
    output = [w for w in word_tokens if not w in stopwords.words('english')]

    return output

def remove_unicode(prompt):
    prompt = prompt.encode("ascii", "ignore")

    return prompt.decode()

def correct_prompt(prompt):
    prompt = TextBlob(prompt)
    prompt = prompt.correct()

    return prompt.string

def text_to_vec(sentence):
    sentences = process_text(sentence)
    temp = sentences

    model = Word2Vec(sentences, vector_size=128, window=5, min_count=1, workers=2)
    vector = model.wv[temp]
    
    return vector