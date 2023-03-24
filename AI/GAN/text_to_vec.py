import tensorflow_hub as hub
from autocorrect import Speller
import string
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#nltk.download('stopwords')
#nltk.download('punkt')

def process_text(prompt):
    prompt = prompt.lower()
    prompt = remove_unicode(prompt)
    prompt = remove_stop_words(prompt)
    return prompt

def remove_stop_words(prompt):
    stop = set(stopwords.words('english') + list(string.punctuation))
    return [i for i in word_tokenize(prompt.lower()) if i not in stop]

def remove_unicode(prompt):
    prompt = prompt.encode("ascii", "ignore")

    return prompt.decode()

def correct_prompt(sentece):
    spell = Speller()

    return spell(sentece)

def text_to_vec(sentence):
    sentence = ' '.join([str(x) for x in sentence])
    
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    embedding = embed([sentence])[0][:128].numpy()

    return embedding