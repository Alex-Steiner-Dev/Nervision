import tensorflow_hub as hub

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#nltk.download('stopwords')
#nltk.download('punkt')

def process_text(prompt):
    prompt = prompt.lower()
    prompt = remove_unicode(prompt)
    #prompt = remove_stop_words(prompt)
    return prompt

def remove_stop_words(prompt):
    word_tokens = word_tokenize(prompt)
    output = [w for w in word_tokens if not w in stopwords.words('english')]

    return output

def remove_unicode(prompt):
    prompt = prompt.encode("ascii", "ignore")

    return prompt.decode()

def text_to_vec(sentence):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    embedding = embed([sentence])[0][:128].numpy()

    return embedding