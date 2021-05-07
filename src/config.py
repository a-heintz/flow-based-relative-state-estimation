import torch.nn as nn
import util

l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()
nll_loss = util.NLLLoss()
softmax = nn.Softmax(dim=1)

#######################
#      CONSTANTS      #
#######################
lin_std                 = 0.5
att_std                 = 0.05
num_inverse_iterations  = 1
SCALE                   = 4 # Scale of image generation process
clamp                   = 2 # clamping constant CINN

#######################
#      Optimizer      #
#######################
lr                      = 1e-4#5 * 10 ** (-5)
betas                   = (0.9, 0.999)
weight_decay            = 1e-6
epsilon                 = 1e-6

#######################
#     Train Args      #
#######################
n_epochs                = 600
batch_size              = 64
num_inv                 = 5
log_dir                 = 'log'
fraction                = 0.1
n_samples               = 100000
workers                 = 4
pretrained_path         = 'models'

#######################
#    Model Inputs     #
#######################
x_dim                   = 3
v_dim                   = 7
r_dim                   = 256
h_dim                   = 128
z_dim                   = 64
L                       = 4
share                   = True
tower_m                 = 2
tower_l                 = 2
loss_coeff              = 1e2
num_levels              = 3
num_steps               = 6

#######################
#     Image Dims      #
#######################
n_context               = 30
n_channels              = 3
img_width               = 64
x_q_zeros_dim           = 4
tower_out_dims          = 16
cont_x_shape            = (batch_size, n_context, n_channels, img_width, img_width)
cont_v_shape            = (batch_size, n_context, v_dim)
query_x_shape           = (batch_size, n_channels, img_width, img_width)
query_v_shape           = (batch_size, v_dim)
query_v_emb_shape       = (v_dim, img_width, img_width)
x_q_zeros_gen_shape     = (batch_size, x_q_zeros_dim, img_width, img_width)
r_shape                 = (batch_size, r_dim, img_width, img_width)
cat_dims                = tower_out_dims**2 + v_dim
rv_q_shape              = (batch_size, cat_dims, img_width, img_width)
resnet_out_shape        = (batch_size, cat_dims*2*2, img_width // 2, img_width // 2)
