from flask import Flask, render_template, request, send_file, session
import torch
import os
import sys

sys.path.append("../AI/GAN/")

from text_to_vec import *
import pyvista as pv
import model
import open3d as o3d
import string
import random

import numpy as np

Generator = model.Generator().cuda()
Autoencoder = model.Autoencoder().cuda()

vertices_path = "../AI/TrainedModels/vertices.pt" 
checkpoint = torch.load(vertices_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

autoencoder_path = "../AI/TrainedModels/autoencoder.pt" 
checkpoint_ae = torch.load(autoencoder_path)
Autoencoder.load_state_dict(checkpoint_ae['autoencoder'])

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

#################################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
sentences = []
ids = []

f = open("../AI/GAN/captions.json")
data = json.load(f)

for i, itObject in enumerate(data):

    if itObject['desc'].split('.')[0].find(".") != -1:
        label = itObject['desc']
    else:
        label = itObject['desc'].split('.')[0]
             
    sentences.append(label)
    ids.append(itObject['id'])

f.close()

def fake(target_sentence):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences + [target_sentence])
    cosine_similarities = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])

    most_similar_index = cosine_similarities.argmax()

    mesh = o3d.io.read_triangle_mesh("../AI/GAN/dataset/" + ids[most_similar_index] + '.obj')
    simplified_mesh = mesh.simplify_quadric_decimation(4096)

    if len(simplified_mesh.vertices) > 4096:
        simplified_mesh = simplified_mesh.simplify_vertex_clustering(.01)

    faces = np.array(simplified_mesh.triangles)
    
    print(np.array(simplified_mesh.vertices).shape)
    return faces
#################################################################################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_model():
    name = generate(request.form['object'])
    url = "static/generations/" + name + "/" + "model.gltf"
    session['url'] = url

    return render_template('generated.html', url=url)

@app.route('/download')
def download_file():
    path = session.get('url')

    return send_file(path, as_attachment=True)

def string_generator(size=12, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def generate(text):
    name = string_generator()
    os.mkdir("static/generations/" + name)

    z = torch.from_numpy(text_to_vec(process_text(correct_prompt(text)))).reshape(1,512,1).repeat(13, 1, 1).cuda().float()

    with torch.no_grad():
        sample = Generator(z).cpu()
        points = sample.numpy()[0]

    vertices = Autoencoder(torch.from_numpy(points).to('cuda')).cpu().detach().numpy()
    vertices = np.array(vertices, dtype=np.float32)
    faces = fake(text)

    mesh = create_mesh(vertices, faces)

    o3d.io.write_triangle_mesh("static/generations/" + name + "/model.obj", mesh)
    
    mesh = pv.read("static/generations/" + name + "/model.obj")
    texture = pv.read_texture('texture.png')

    mesh.texture_map_to_plane(inplace=True)

    p = pv.Plotter()
    p.add_mesh(mesh, texture=texture)
    p.export_gltf("static/generations/" + name + "/model.gltf")

    return name

def create_mesh(vertices, faces):
    vertices = np.array(vertices)
    faces = np.array(faces)

    mesh = o3d.geometry.TriangleMesh()
    
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()

    return mesh


if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
