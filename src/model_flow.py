import config as c
import torch
import torch.optim
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from iResNet import *
from tower import TowerRepresentation
from torch.distributions import Normal, kl_divergence
from gqn_utils import *

class Generator(nn.Module):
    def __init__(self, device, L=12, r_dim=c.r_dim, z_dim=c.z_dim, h_dim=c.h_dim, v_dim=c.v_dim):
        super().__init__()
        self.r_dim = r_dim
        self.device = device

        iResNet_in_shape = (c.n_channels + c.n_channels + v_dim, c.img_width, c.img_width)
        resnet_channels = c.img_width
        #self.phi_r = conv_iResNet((c.n_channels, c.img_width, c.img_width), [8], [1], [64],
        #                         init_ds=2, density_estimation=True, actnorm=True, numSeriesTerms = 5, nonlin="relu")
        #self.phi_x = conv_iResNet((c.n_channels, c.img_width, c.img_width), [8], [1], [64],
        #                         init_ds=2, density_estimation=True, actnorm=True, numSeriesTerms = 5, nonlin="relu")
        #self.phi_v = conv_iResNet((v_dim, c.img_width, c.img_width), [8], [1], [64],
        #                         init_ds=2, density_estimation=True, actnorm=True, numSeriesTerms = 5, nonlin="relu")
        self.phi = conv_iResNet(iResNet_in_shape, [4], [1], [1024],
                                 init_ds=2, density_estimation=True, actnorm=True, numSeriesTerms = 5, nonlin="relu")
        self.gamma = conv_iResNet(iResNet_in_shape, [4], [1], [1024],
                                 init_ds=2, density_estimation=True, actnorm=True, numSeriesTerms = 5, nonlin="relu")

        #self.phi_v_est = conv_iResNet(c.n_channels, [16], [1], [64],
        #                         init_ds=2, density_estimation=True, actnorm=True, numSeriesTerms = 1)

        #self.phi = multiscale_conv_iResNet(iResNet_in_shape, [16, 16], [1, 1], [128, 128],
        #                         True, 0, 0.9, True, None, 1, 5, 5, True, True)
        #self.gamma = multiscale_conv_iResNet(iResNet_in_shape, [16, 16], [1, 1], [128, 128],
        #                         True, 0, 0.9, True, None, 1, 5, 5, True, True)

    def bits_per_dim(self, logpx, inputs):
        return -logpx / float(np.log(2.) * np.prod(inputs.shape[1:])) + 8.

    def forward(self, x, v_q, r, ignore_logdet=False):
        batch_size = r.size(0)
        h, w = c.img_width, c.img_width
        SCALE = 1
        v = v_q.view(batch_size, -1, 1, 1).repeat(1, 1, h // SCALE, w // SCALE)
        r = r.repeat(1, 1, h // r.shape[2], w // r.shape[3])
        v_comp = torch.zeros_like(v)#((batch_size, *c.query_v_emb_shape)).to(x.get_device())

        phi_in = torch.cat((r, x, v), dim = 1)
        z_phi, _, tr_phi = self.phi(phi_in, ignore_logdet)

        psi_in = torch.cat((r, x, v_comp), dim = 1)
        z_psi, _, tr_psi = self.gamma(psi_in, ignore_logdet)

        z_mu_phi, z_std_phi = z_phi.mean().item(), z_phi.std().item()
        z_mu_psi, z_std_psi = z_psi.mean().item(), z_psi.std().item()

        phi_prior = Normal(z_mu_phi, z_std_phi)
        psi_prior = Normal(z_mu_psi, z_std_psi)

        logpz_phi = psi_prior.log_prob(z_phi.view(z_phi.size(0), -1)).sum(dim=1)
        logpz_psi = phi_prior.log_prob(z_psi.view(z_psi.size(0), -1)).sum(dim=1)

        logpx_phi = (logpz_phi + tr_phi).mean()
        logpx_psi = (logpz_psi + tr_psi).mean()

        nll_phi = self.bits_per_dim(logpx_phi, Variable(phi_in, requires_grad=True)).mean()
        nll_psi = self.bits_per_dim(logpx_psi, Variable(psi_in, requires_grad=True)).mean()

        phi_in_re = self.phi.inverse(z_psi, c.num_inv)
        v_re = torch.mean(phi_in_re[:,c.n_channels*2:,:,:], dim=(2,3))
        v_loss = c.mse_loss(v_q, v_re)

        nll = nll_phi + nll_psi
        z_loss = v_loss

        return nll, z_loss

    def inverse_state_est(self, r, x, ignore_logdet=False):
        batch_size = r.size(0)
        h, w = c.img_width, c.img_width
        SCALE = 1
        r = r.repeat(1, 1, h // r.shape[2], w // r.shape[3])
        z_r, logpz_r, tr_r = self.phi_r(r, ignore_logdet)
        z_x, logpz_x, tr_x = self.phi_x(x, ignore_logdet)
        v_comp = torch.zeros((batch_size, *c.query_v_emb_shape))#.cpu()#.to(z_x.get_device())
        inp_target = torch.cat((z_r.view(r.shape), z_x.view(x.shape), v_comp), dim = 1)
        z_psi, logpz_psi, tr_psi = self.gamma(inp_target, ignore_logdet)
        z_v_re = self.phi.inverse(z_psi, c.num_inv)
        r_re = self.phi_r.inverse(z_v_re[:,:c.n_channels,:,:].view(z_r.shape), c.num_inv)
        x_re = self.phi_x.inverse(z_v_re[:,c.n_channels:c.n_channels * 2,:,:].view(z_x.shape), c.num_inv)
        v_re = self.phi_v.inverse(z_v_re[:,c.n_channels * 2:,:,:].view((batch_size, *self.phi_v.final_shape)), c.num_inv)
        v_re = torch.mean(v_re, dim=(2,3))
        return x_re, r_re, v_re

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
        r = torch.sigmoid(r)
        return r

    def forward(self, x_c, v_c, x_q, v_q, ignore_logdet):
        r = self.representation(x_c, v_c)
        nll, z_loss = self.generator(x_q, v_q, r, ignore_logdet)
        return nll, z_loss

    def inverse_state_est(self, x_c, v_c, x, ignore_logdet=True):
        r = self.representation(x_c, v_c)
        x_re, r_re, v_re = self.generator.inverse_state_est(r, x, ignore_logdet=ignore_logdet)
        return x_re, r_re, v_re

class Model():
    def __init__(self, combined_model, optimizer):
        super().__init__()
        self.combined_model     = combined_model
        self.optimizer          = optimizer

def get_model(device):
    combined_model              = Architecture(device)
    combined_model              = combined_model.to(device)
    combined_model              = nn.DataParallel(combined_model, device_ids=[0])
    params_trainable            = combined_model.parameters()
    optimizer                   = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=c.epsilon, weight_decay=c.weight_decay)
    model                       = Model(combined_model, optimizer)

    return model, optimizer
