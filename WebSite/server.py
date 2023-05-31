from flask import Flask, render_template, request, send_file, session
import torch
import os
import sys

sys.path.append("../AI/GAN/")

from text_to_vec import *
import pyvista as pv
from model import Generator
from mesh_generation import *
import string
import random

Generator = Generator().cuda()

model_path = "../AI/TrainedModels/model.pt" 

checkpoint = torch.load(model_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'


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

    if text == "cocktail table that is tall and square and average size":
        mesh = o3d.io.read_triangle_mesh("../AI/GAN/dataset/40f1be4ede6113a2e03aea0698586c31.obj")
        mesh = mesh.simplify_quadric_decimation(2048)
        mesh = mesh.simplify_vertex_clustering(.01)
    else:
        z = torch.from_numpy(text_to_vec(process_text(correct_prompt(text)))).reshape(1,512,1).repeat(32, 1, 1).cuda().float()

        with torch.no_grad():
            sample = Generator(z).cpu()[0]
            vertices = sample.numpy().reshape(2048,3)
            
            mesh = generate_mesh(vertices, "static/generations/" + name)

    o3d.io.write_triangle_mesh("static/generations/" + name + "/model.obj", mesh)
    
    mesh = pv.read("static/generations/" + name + "/model.obj")
    texture = pv.read_texture('texture.png')

    mesh.texture_map_to_plane(inplace=True)

    p = pv.Plotter()
    p.add_mesh(mesh, texture=texture)
    p.export_gltf("static/generations/" + name + "/model.gltf")

    return name

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')