import torch
import torchvision.models as models
from model import *

from text_to_vec import *

import numpy as np
import pyvista as pv

target_model = Generator().to('cuda')

pretrained_model1 = Generator().to('cuda')
pretrained_model2 = Generator().to('cuda')

checkpoint_model1 = torch.load('../TrainedModels/model_0.pt')
checkpoint_model2 = torch.load('../TrainedModels/model_1.pt')

pretrained_model1.load_state_dict(checkpoint_model1['G_state_dict'])
pretrained_model2.load_state_dict(checkpoint_model2['G_state_dict'])

combined_state_dict = target_model.state_dict()

for name, param in pretrained_model1.named_parameters():
    combined_state_dict[name] = param.data

for name, param in pretrained_model2.named_parameters():
    combined_state_dict[name] = param.data

target_model.load_state_dict(combined_state_dict)

z = torch.from_numpy(text_to_vec(process_text(correct_prompt("cocktail table that is tall and square and average sizel"))).astype(np.float64)).reshape(1,512, 1).cuda().float()

with torch.no_grad():
    sample = target_model(z).cpu()

    vertices = sample.numpy()[0]

    pv.PolyData(vertices).plot()