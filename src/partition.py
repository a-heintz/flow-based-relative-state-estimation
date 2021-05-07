import random
import torch
import numpy as np
import config as c

def image_resize(T, shape):
    # input shape [3, 200, 120]
    #T = T.unsqueeze(0)
    T = torch.nn.functional.interpolate(T, size=shape, mode='bicubic')
    #T = T.squeeze(0)
    # output shape [3, 100, 80]
    return T

def partition(images, viewpoints, d):
    """
    Partition batch into context and query sets.
    :param images
    :param viewpoints
    :return: context images, context viewpoint, query image, query viewpoint
    """
    # Maximum number of context points to use
    _, b, m, *x_dims = images.shape
    _, b, m, *v_dims = viewpoints.shape
    _, b, m, *d_dims = d.shape

    #print(images.shape, viewpoints.shape, d.shape)

    # "Squeeze" the batch dimension
    images = images.view((-1, m, *x_dims))
    viewpoints = viewpoints.view((-1, m, *v_dims))

    d = d.view((-1, m, *d_dims))

    # Sample random number of views
    n_context = c.n_context#random.randint(2, m - 1)
    indices = random.sample([i for i in range(m)], n_context)

    # Partition into context and query sets
    context_idx, query_idx = indices[:-1], indices[-1]

    x_c, v_c, d_c = images[:, context_idx], viewpoints[:, context_idx], d[:, context_idx]
    x_q, v_q, d_q = images[:, query_idx], viewpoints[:, query_idx], d[:, query_idx]

    '''
    x_c_ = torch.zeros(c.batch_size, x_c.shape[1], c.n_channels, c.img_width, c.img_width)
    for b in range(c.batch_size):
        for cont in range(x_c[b].shape[0]):
            x_c_[b,cont] = image_resize(x_c[b,cont].view(1, *x_c.shape[2:]), (c.img_width, c.img_width)).view(-1, c.img_width, c.img_width)
    x_c = x_c_
    x_q = image_resize(x_q, (c.img_width, c.img_width))
    '''

    return x_c, v_c, x_q, v_q, d_c, d_q
