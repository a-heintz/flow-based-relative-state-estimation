import config as c
import torch
import torch.optim
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from iResNet_cond import *
from tower import TowerRepresentation
from torch.distributions import Normal, kl_divergence
from gqn_utils import *
from glow.glow import Glow


class Generator(nn.Module):
    def __init__(self, device, L=12, r_dim=c.r_dim, z_dim=c.z_dim, h_dim=c.h_dim, v_dim=c.v_dim):
        super().__init__()
        self.device = device

        phi_shape = (v_dim, c.img_width, c.img_width)
        self.phi = Glow(in_channels = c.v_dim * 4,
                        cond_channels = c.n_channels * 4 + c.n_channels * 4,
                        num_channels = 256,
                        num_levels = c.num_levels,
                        num_steps = c.num_steps,
                        mode = 'sketch')
    def logit(self, u, eps = 1e-7):
        return torch.log( (u - eps) / (1 - (u - eps)) )

    def bits_per_dim(self, logpx, inputs):
        return -logpx / float(np.log(2.) * np.prod(inputs.shape[1:])) + 8.

    def pre_process(self, v):
        v[:,:3] += 1.
        v[:,:3] /= 2.

        v[:,3:] += 20.
        v[:,3:] /= 40.

        return v

    def post_process(self, v):
        v[:,:3] *= 2.
        v[:,:3] -= 1.

        v[:,3:] *= 40.
        v[:,3:] -= 20.

        return v

    def forward(self, x, v_q, r, d, ignore_logdet=False):
        batch_size = r.size(0)
        h, w = c.img_width, c.img_width
        SCALE = 1


        #v_q_ = c.softmax(v_q)

        #v = distributions.Normal(self.prior_mu, torch.exp(self.prior_logstd))

        #v = v_q.view(batch_size, -1, 1, 1).repeat(1, 1, h // SCALE, w // SCALE)

        r = r.repeat(1, 1, h // r.shape[2], w // r.shape[3])

        phi_cond = torch.cat((r, x), dim = 1)

        z, ldj = self.phi(d, phi_cond, reverse=False)



        nll = c.nll_loss(z, ldj)
        #nll = ldj.mean() * 0
        #v_re = self.sample(x, r)

        #v_q = self.post_process(v_q)
        #v_re = torch.mean(v_re, dim=(2,3))
        z_loss = nll * 0
        #z_loss = c.mse_loss(v_re, v_q)

        return nll, z_loss

    def test(self, x, v_q, r, d, ignore_logdet=False):
        batch_size = r.size(0)
        h, w = c.img_width, c.img_width
        SCALE = 1

        #v_q = self.pre_process(v_q)

        #v_q_ = c.log_softmax(v_q)

        #v = v_q.view(batch_size, -1, 1, 1).repeat(1, 1, h // SCALE, w // SCALE)

        r = r.repeat(1, 1, h // r.shape[2], w // r.shape[3])

        phi_cond = torch.cat((r, x), dim = 1)

        z, ldj = self.phi(d, phi_cond, reverse=False)

        v_re, _ = self.phi(z, phi_cond, reverse=True)



        v_q_ = v_re
        #v_re = torch.mean(v_re, dim=(2,3))

        return v_re, v_q_

    def sample(self, x, r, sigma=1):
        batch_size = r.size(0)
        h, w = c.img_width, c.img_width
        SCALE = 1

        r = r.repeat(1, 1, h // r.shape[2], w // r.shape[3])
        phi_cond = torch.cat((r, x), dim = 1)

        z = sigma * torch.cuda.FloatTensor(batch_size, c.v_dim, h, w).normal_().to(self.device)
        v, _ = self.phi(z, phi_cond, reverse=True)
        #v = torch.mean(v, dim=(2,3))

        #v = self.post_process(v)



        return v


class Architecture(nn.Module):
    def __init__(self, device, n_channels=c.n_channels, v_dim=c.v_dim, r_dim=c.r_dim):
        super().__init__()
        self.device = device
        self.tower = TowerRepresentation(c.n_channels, v_dim, r_dim, pool=True)
        self.generator = Generator(device)

    def representation(self, x_c, v_c):
        batch_size, n_views, _, h, w = x_c.shape
        _, _, *x_dims = x_c.shape
        _, _, *v_dims = v_c.shape
        x = x_c.view((-1, *x_dims))
        v = v_c.view((-1, *v_dims))
        phi = self.tower(x, v)
        _, *phi_dims = phi.shape
        phi = phi.view((batch_size, n_views, *phi_dims))
        r = torch.sum(phi, dim=1)
        return r

    def forward(self, x_c, v_c, x_q, v_q, d_c, d_q, ignore_logdet):
        r = self.representation(x_c, d_c)
        nll, z_loss = self.generator(x_q, v_q, r, d_q, ignore_logdet)
        return nll, z_loss

    def test(self, x_c, v_c, x_q, v_q, d_c, d_q, ignore_logdet=True):
        r = self.representation(x_c, d_c)
        v_re, v_q_ = self.generator.test(x_q, v_q, r, d_q)
        return v_re, v_q_

    def sample(self, x_c, v_c, x, d_c, sigma=1):
        r = self.representation(x_c, d_c)
        v_sample = self.generator.sample(x, r, sigma)
        return v_sample

class Model():
    def __init__(self, combined_model, optimizer):
        super().__init__()
        self.combined_model     = combined_model
        self.optimizer          = optimizer

def get_model(device):
    combined_model              = Architecture(device)
    combined_model              = combined_model.to(device)
    combined_model              = nn.DataParallel(combined_model, device_ids=[1])
    params_trainable            = combined_model.parameters()
    optimizer                   = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=c.epsilon, weight_decay=c.weight_decay)
    model                       = Model(combined_model, optimizer)

    return model, optimizer
