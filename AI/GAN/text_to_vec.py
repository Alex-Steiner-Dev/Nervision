import torch
from sentence_transformers import SentenceTransformer

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
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    sentences = [sentence]

    embeddings = model.encode(sentences, convert_to_tensor=True)

    projector = torch.nn.Linear(768, 128).cuda()
    embeddings = projector(embeddings).cuda()
    embeddings = embeddings.cpu().detach().numpy()

    return embeddings