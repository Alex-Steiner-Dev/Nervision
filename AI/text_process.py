from autocorrect import Speller
import re

import nltk
import nltk.corpus
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')

spell = Speller(lang='en')

stop_words = set(stopwords.words('english'))
stop_words.remove("with")
stop_words.add("generate")
stop_words.add("create")

stemmer = PorterStemmer()

def process_text(prompt):
    prompt = prompt.lower()
    prompt = correct_prompt(prompt)
    prompt = remove_unicode(prompt)
    prompt = remove_stop_words(prompt)
    #prompt = stemming(prompt)
    prompt = part_of_speech(prompt)

    return prompt

def correct_prompt(prompt):
    return spell(prompt)

def remove_unicode(prompt):
    return re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", prompt)

def remove_stop_words(prompt):
    output = ""
  
    word_tokens = word_tokenize(prompt)
    output = [w for w in word_tokens if not w in stop_words]
  
    return output

def stemming(prompt):
    output = []
    for x in prompt:
        output.append(stemmer.stem(x))
    return output

def part_of_speech(prompt):
    pos = nltk.pos_tag(prompt)

    return pos