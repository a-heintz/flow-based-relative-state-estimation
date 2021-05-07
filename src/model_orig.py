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

        self.device = device
        x_dim = c.n_channels
        self.L = L
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.SCALE = 4
        kwargs = dict(kernel_size=5, stride=1, padding=2)
        inference_args = dict(in_channels=v_dim + r_dim + x_dim + h_dim, out_channels=h_dim, **kwargs)
        generator_args = dict(in_channels=v_dim + r_dim + z_dim, out_channels=h_dim, **kwargs)
        self.inference_core = Conv2dLSTMCell(**inference_args)
        self.generator_core = Conv2dLSTMCell(**generator_args)
        # Inference, prior
        self.posterior_density = nn.Conv2d(h_dim, 2*z_dim, **kwargs)
        self.prior_density     = nn.Conv2d(h_dim, 2*z_dim, **kwargs)
        # Generative density
        self.observation_density = nn.Conv2d(h_dim, x_dim, kernel_size=1, stride=1, padding=0)
        # Up/down-sampling primitives
        self.upsample   = nn.ConvTranspose2d(h_dim, h_dim, kernel_size=self.SCALE, stride=self.SCALE, padding=0, bias=False)
        self.downsample = nn.Conv2d(x_dim, x_dim, kernel_size=self.SCALE, stride=self.SCALE, padding=0, bias=False)

        descrim_dim = (c.n_channels + c.tower_out_dims + c.v_dim, c.img_width, c.img_width)
        resnet_channels = c.img_width
        self.descriminator = conv_iResNet(descrim_dim, [1], [1], [resnet_channels],
                                 init_ds=2, density_estimation=True, actnorm=True)

    def bits_per_dim(self, logpx, inputs):
        return -logpx / float(np.log(2.) * np.prod(inputs.shape[1:])) + 8.
        
    def forward(self, x_q, v_q, r, ignore_logdet=False):

        batch_size = r.shape[0]
        # GQN CORE
        kl = 0
        # Downsample x, upsample v and r
        x = self.downsample(x_q)
        v = v_q.view(batch_size, -1, 1, 1).repeat(1, 1, c.img_width // self.SCALE, c.img_width // self.SCALE)
        if r.size(2) != c.img_width // self.SCALE:
            r = r.repeat(1, 1, c.img_width // self.SCALE, c.img_width // self.SCALE)
        # Reset hidden and cell state
        hidden_i = x.new_zeros((batch_size, self.h_dim, c.img_width // self.SCALE, c.img_width // self.SCALE))
        cell_i   = x.new_zeros((batch_size, self.h_dim, c.img_width // self.SCALE, c.img_width // self.SCALE))
        hidden_g = x.new_zeros((batch_size, self.h_dim, c.img_width // self.SCALE, c.img_width // self.SCALE))
        cell_g   = x.new_zeros((batch_size, self.h_dim, c.img_width // self.SCALE, c.img_width // self.SCALE))
        u = x.new_zeros((batch_size, self.h_dim, c.img_width, c.img_width))

        for l in range(self.L):
            # Prior factor (eta π network)
            p_mu, p_std = torch.chunk(self.prior_density(hidden_g), 2, dim=1)
            prior_distribution = Normal(p_mu, F.softplus(p_std))
            # Inference state update
            hidden_i, cell_i = self.inference_core(torch.cat([hidden_g, x, v, r], dim=1), [hidden_i, cell_i])
            # Posterior factor (eta e network)
            q_mu, q_std = torch.chunk(self.posterior_density(hidden_i), 2, dim=1)
            posterior_distribution = Normal(q_mu, F.softplus(q_std))
            # Posterior sample
            z = posterior_distribution.rsample()
            # Generator state update
            hidden_g, cell_g = self.generator_core(torch.cat([z, v, r], dim=1), [hidden_g, cell_g])
            # Calculate u
            u = self.upsample(hidden_g) + u
            # Calculate KL-divergence
            kl += kl_divergence(posterior_distribution, prior_distribution)
        x_mu = self.observation_density(u)
        
        x_mu_gqn = torch.sigmoid(x_mu)
        
        
        # invertible descriminator CORE
        r_ = r.view(batch_size, *c.r_shape[1:])
        v_ = v_q.view(batch_size, -1, 1, 1).repeat(1, 1, c.img_width, c.img_width)
        descriminator_input = torch.cat((x_mu_gqn, r_, v_), dim = 1)
        descriminator_output, logpz, tr = self.descriminator(descriminator_input, ignore_logdet)
        descriminator_output = descriminator_output.view(batch_size, -1, c.img_width, c.img_width)
        x_mu = descriminator_output[:,:c.n_channels,:,:]
        r_gen = descriminator_output[:,c.n_channels:c.n_channels+c.tower_out_dims,:,:].view(batch_size, *c.r_shape[1:])
        z_gen = descriminator_output[:,c.n_channels+c.tower_out_dims:,:,:]

        

        r_loss = c.mse_loss(r_gen, r_)
        z_loss = c.mse_loss(z_gen, 0.5 * torch.ones_like(z_gen))

        
        
        x_mu = torch.sigmoid(x_mu)
        
        #r_loss = Variable(torch.zeros(1).type(torch.FloatTensor), requires_grad = True).cuda()
        #z_loss = Variable(torch.zeros(1).type(torch.FloatTensor), requires_grad = True).cuda()
        
        return x_mu, x_mu_gqn, kl, r_loss, z_loss, self.bits_per_dim(logpz + tr, descriminator_input).mean()

    def gqn_sample(self, v_q, r, ignore_logdet=True):
        batch_size = r.size(0)
        h, w = c.img_width, c.img_width
        SCALE = self.SCALE
        # Increase dimensions
        v = v_q.view(batch_size, -1, 1, 1).repeat(1, 1, h // SCALE, w // SCALE)
        if r.size(2) != h // SCALE:
            r = r.repeat(1, 1, h // SCALE, w // SCALE)
        # Reset hidden and cell state for generator
        hidden_g = v.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))
        cell_g = v.new_zeros((batch_size, self.h_dim, h // SCALE, w // SCALE))
        u = v.new_zeros((batch_size, self.h_dim, h, w))
        for _ in range(self.L):
            p_mu, p_log_std = torch.chunk(self.prior_density(hidden_g), 2, dim=1)
            prior_distribution = Normal(p_mu, F.softplus(p_log_std))
            # Prior sample
            z = prior_distribution.sample()
            # Calculate u
            hidden_g, cell_g = self.generator_core(torch.cat([z, v, r], dim=1), [hidden_g, cell_g])
            u = self.upsample(hidden_g) + u
        x_mu = self.observation_density(u)

        # invertible descriminator CORE
        r_ = r.view(batch_size, *c.r_shape[1:])
        v_ = v_q.view(batch_size, -1, 1, 1).repeat(1, 1, c.img_width, c.img_width)
        descriminator_input = torch.cat((x_mu, r_, v_), dim = 1)
        descriminator_output, _, _ = self.descriminator(descriminator_input, ignore_logdet)
        descriminator_output = descriminator_output.view(batch_size, -1, c.img_width, c.img_width)
        x_mu = descriminator_output[:,:c.n_channels,:,:]
        r_gen = descriminator_output[:,c.n_channels:c.n_channels+c.r_dim,:,:]
        z_gen = descriminator_output[:,c.n_channels+c.r_dim:,:,:]

        x_mu = torch.sigmoid(x_mu)

        return x_mu

    def sample(self, x_c, v_c, v_q, ignore_logdet=False):
        return

    def inverse_sigmoid(self, u):
        return torch.log( u / (1-u) )

    def inverse_sample(self, x_mu, x_c, v_c):
        return

class Architecture(nn.Module):
    def __init__(self, device, n_channels=c.n_channels, v_dim=c.v_dim, r_dim=c.r_dim):
        super().__init__()

        self.device             = device
        self.tower     = TowerRepresentation(c.n_channels, v_dim, r_dim, pool=True)

        #self.tower              = TowerRepresentation(c.n_channels, c.img_width * c.img_width, c.v_dim, 256, pool=False)
        self.generator          = Generator(device)
    '''
    def representation(self, x_c, v_c):
        b, m, *x_dims           = x_c.shape
        _, _, *v_dims           = v_c.shape
        x                       = x_c.view((-1, *x_dims))
        v                       = v_c.view((-1, *v_dims))
        phi                     = self.tower(x, v)
        _, *phi_dims            = phi.shape
        phi                     = phi.view((b, m, *phi_dims))
        r                       = torch.sum(phi, dim=1)
        r                       = r.view(b, *c.r_shape[1:])
        return r
    '''
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

    def forward(self, x_c, v_c, x_q, v_q, ignore_logdet=False):
        r = self.representation(x_c, v_c)
        x_mu, x_mu_gqn, kl, r_loss, z_loss, logpx = self.generator(x_q, v_q, r, ignore_logdet)
        return x_mu, x_mu_gqn, kl, r_loss, z_loss, logpx

    def gqn_sample(self, x_c, v_c, v_q, ignore_logdet=True):
        r                       = self.representation(x_c, v_c)
        x_mu                    = self.generator.gqn_sample(v_q, r)
        return x_mu

    def inverse_sample(self, x_mu, x_c, v_c):
        return

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
