import torch
from torch.autograd import Variable, grad

class GradientPenalty:
    def __init__(self, lambdaGP, gamma=1, vertex_num=2500, device=torch.device('cpu')):
        self.lambdaGP = lambdaGP
        self.gamma = gamma
        self.vertex_num = vertex_num
        self.device = device

    def __call__(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)

        fake_data = fake_data[:batch_size]
        
        alpha = torch.rand(batch_size, 1, 1, requires_grad=True).to(self.device)
        interpolates = real_data + alpha * (fake_data - real_data)
       
        disc_interpolates, _ = netD(interpolates)
      
        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0].contiguous().view(batch_size,-1)
                         
        gradient_penalty = (((gradients.norm(2, dim=1) - self.gamma) / self.gamma) ** 2).mean() * self.lambdaGP

        return gradient_penalty