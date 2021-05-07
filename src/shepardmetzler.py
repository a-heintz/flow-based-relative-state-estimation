import os, gzip
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import fnmatch


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


class ShepardMetzler(Dataset):
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
    def __init__(self, root_dir, train=True, transform=None, fraction=1.0, target_transform=transform_viewpoint):
        super(ShepardMetzler, self).__init__()
        assert fraction > 0.0 and fraction <= 1.0
        if train:
            self.data_loc = root_dir + '/train'
        else:
            self.data_loc = root_dir + '/test'
        self.paths = {}
        self.df = pd.read_csv(self.data_loc + '/fin_numbers.txt', header = None)

        for _, t in enumerate(self.df[0]):
            if len(str(t)) == 1:
                self.paths[t] = self.data_loc + '/gqn_scene_00000' + str(t) + '/'
            elif len(str(t)) < 3:
                self.paths[t] = self.data_loc + '/gqn_scene_0000' + str(t) + '/'
            else:
                self.paths[t] = self.data_loc + '/gqn_scene_000' + str(t) + '/'


        self.records = []
        # root directory is up to shepard_metzler_5_parts
        # now iterate through folders getting all .pt.gz files
        for _, _, files in os.walk(self.data_loc):
          for filename in files:
              if fnmatch.fnmatch(filename, '*.gz'):
                    self.records.append(str(filename))
        self.records = sorted(self.records)
        self.records = self.records[:int(len(self.records)*fraction)]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        num = int(self.records[idx][:3])
        scene_path = os.path.join(self.paths[num], self.records[idx])
        with gzip.open(scene_path, "r") as f:
            data = torch.load(f)
            images, viewpoints = list(zip(*data))

        images = np.stack(images)
        viewpoints = np.stack(viewpoints)

        # uint8 -> float32
        images = images.transpose(0, 1, 4, 2, 3)
        images = torch.FloatTensor(images)/255
        
        if self.transform:
            images = self.transform(images)

        viewpoints = torch.FloatTensor(viewpoints)
        if self.target_transform:
            viewpoints = self.target_transform(viewpoints)

        return images, viewpoints