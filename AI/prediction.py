import sys

sys.path.append('GAN/')

import torch
from utils import *
from model import net_G
import pyvista as pv
import numpy as np

from keras.models import load_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def predict():
    generator_path =  'TrainedModels/G.pth'
    ae_path = 'TrainedModels/autoencoder.h5'

    ae = load_model(ae_path)
    generator = net_G()


    if not torch.cuda.is_available():
        generator.load_state_dict(torch.load(generator_path, map_location={'cuda:0': 'cpu'}))
    else:
        generator.load_state_dict(torch.load(generator_path))

    generator.to(params.device)

    os.system("clear")

    z = generateZ(1)

    fake = generator(z)
    samples = fake.unsqueeze(dim=0).detach().cpu().numpy()
    reconstruction = ae.predict(samples).reshape(samples.shape)

    SavePloat_Voxels(reconstruction, "images/", 1)  

predict()