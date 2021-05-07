import os, gzip
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from scipy.stats import multivariate_normal
import fnmatch
import matplotlib.pyplot as plt
import config as c
import tqdm

def transform_viewpoint(v):
    """
    Transforms the viewpoint vector into a consistent
    representation
    """

    w, z = torch.split(v, 3, dim=-1)
    y, p = torch.split(z, 1, dim=-1)

    # position, [yaw, pitch]
    view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    v_hat = torch.cat(view_vector, dim=-1)

    return v_hat


class CircularOrbit(Dataset):
    """
    Shepart Metzler mental rotation task
    dataset. Based on the dataset provided
    in the GQN paper. Either 5-parts or
    7-parts.
    :param root_dir: location of data on disc
    :param train: whether to use train of test set
    :param transform: transform on images
    :param fraction: fraction of dataset to use
    :param target_transform: transform on viewpoints
    """
    def __init__(self, root_dir='../../circular_orbit', n_samples = 40000):
        super(CircularOrbit, self).__init__()
        self.orbits_dir_path = os.path.join(root_dir, "images")
        self.orbits_dir = os.listdir(self.orbits_dir_path)
        self.idxs = list(range(n_samples))
        self.orbits_pos = np.load(os.path.join(root_dir, "state_files", "orbits_positions.npy"))
        self.orbits_att = np.load(os.path.join(root_dir, "state_files", "orbits_attitudes.npy"))
        self.num_imgs = 40
        self.num_select_imgs = c.n_context
        self.lin_std = c.lin_std
        self.att_std = c.att_std
        self.grid_dim = c.img_width

        self.orbits_arr = []
        self.rand_image_idxs_arr = []
        self.imgs = []

        for idx in tqdm.tqdm(self.idxs):

            orbit = np.random.choice(self.orbits_dir)

            rand_image_idxs = np.random.choice(list(range(self.num_imgs)), self.num_select_imgs, replace=False)

            rand_image_idxs = np.sort(rand_image_idxs)

            self.orbits_arr.append(orbit)
            self.rand_image_idxs_arr.append(rand_image_idxs)

    def __len__(self):
        return len(self.idxs)

    def create_state_density(self, v_q, grid_dim, std, xmin, xmax, n_states_min, n_states_max):
        maps = np.zeros((n_states_max - n_states_min, grid_dim, grid_dim))
        for i in range(n_states_max - n_states_min):
            mu_x = v_q[i]
            if n_states_max == 7:
                mu_x = v_q[n_states_min + i]
            variance_x = std**2
            x = np.linspace(xmin,xmax,grid_dim)
            rv = multivariate_normal.pdf(x, mu_x, variance_x)
            map_ = rv.reshape(grid_dim, 1).repeat(grid_dim, axis=1)
            maps[i, :, :] = map_
        return maps

    def __getitem__(self, idx):

        orbit = self.orbits_arr[idx]
        rand_image_idxs = self.rand_image_idxs_arr[idx]

        viewpoints = []
        densities = []
        imgs = []
        
        for i in rand_image_idxs:
            img_path = os.path.join(self.orbits_dir_path, orbit, str(i).zfill(6) + '.png')
            img = plt.imread(img_path)[:,:,:3]
            imgs.append(img)

            viewpoint = np.append(self.orbits_pos[int(orbit), i], self.orbits_att[int(orbit), i])

            lins = self.create_state_density(viewpoint, self.grid_dim, self.lin_std, -20, 20, 0, 3)
            atts = self.create_state_density(viewpoint, self.grid_dim, self.att_std, -1, 1, 3, 7)
            atts /= 100.
            density = np.vstack((lins, atts))

            densities.append(density)
            viewpoints.append(viewpoint)

        imgs = np.stack(imgs).transpose(0,3,1,2)
        imgs = torch.FloatTensor(imgs).view(1, *imgs.shape)

        viewpoints = np.stack(viewpoints)
        viewpoints = torch.FloatTensor(viewpoints).view(1, *viewpoints.shape)

        densities = np.stack(densities)
        densities = torch.FloatTensor(densities).view(1, *densities.shape)

        return imgs, viewpoints, densities
