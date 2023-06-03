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
mesh = o3d.io.read_triangle_mesh("../AI/GAN/dataset/tractor.obj")
simplified_mesh = mesh.simplify_quadric_decimation(4096)

if len(simplified_mesh.vertices) > 4096:
    simplified_mesh = simplified_mesh.simplify_vertex_clustering(.0005)

faces = np.array(simplified_mesh.triangles)
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

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    return mesh


if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')