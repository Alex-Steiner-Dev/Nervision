import torch
from utils import *
from model import net_G

import numpy as np
import params

def test():
    pretrained_file_path_G =  'models/G.pth'

    G = net_G()

    if not torch.cuda.is_available():
        G.load_state_dict(torch.load(pretrained_file_path_G, map_location={'cuda:0': 'cpu'}))
    else:
        G.load_state_dict(torch.load(pretrained_file_path_G))

    G.to(params.device)

    z = generateZ(1)

    fake = G(z)
    samples = fake.unsqueeze(dim=0).detach().cpu().numpy()
    SavePloat_Voxels(samples, "models/", 'tester_' + str(0))  

test()