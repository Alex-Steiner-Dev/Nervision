import trimesh
import numpy as np
import open3d as o3d
import json
from mesh_generation import *
from text_to_vec import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sentences = []
ids = []
target_sentence = "desk"

f = open("captions.json")
data = json.load(f)

for i, itObject in enumerate(data):

    if itObject['desc'].split('.')[0].find(".") != -1:
        label = itObject['desc']
    else:
        label = itObject['desc'].split('.')[0]
             
    sentences.append(label)
    ids.append(itObject['mid'])

f.close()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences + [target_sentence])
cosine_similarities = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])

most_similar_index = cosine_similarities.argmax()
most_similar_sentence = sentences[most_similar_index]

print(most_similar_sentence, most_similar_index)

mesh = trimesh.load('dataset/' + ids[most_similar_index] + '.obj', force="mesh")

vertices, _ = trimesh.sample.sample_surface(mesh, count=100000)
points = np.array(vertices, dtype=np.float32)

mesh = generate_mesh(points)
o3d.visualization.draw_geometries([mesh])