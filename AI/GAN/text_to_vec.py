from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

def text_to_vec(input_txt):
    input_vec = model.encode(input_txt)
    return(input_vec)

print(text_to_vec("An airplane with five wings").shape)