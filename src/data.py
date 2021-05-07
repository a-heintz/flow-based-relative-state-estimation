import config as c
import torch
from shepardmetzler import ShepardMetzler
from circularorbit_dataset import CircularOrbit
from torch.utils.data import DataLoader

def get_data():
    cuda = torch.cuda.is_available()
    #######################
    #     Load Dataset    #
    #######################

    #dataset = CircularOrbit

    #train_dataset               = ShepardMetzler(root_dir=c.data_dir, fraction=c.fraction)
    #valid_dataset               = ShepardMetzler(root_dir=c.data_dir, fraction=c.fraction, train=False)

    train_dataset               = CircularOrbit(n_samples = c.n_samples)
    valid_dataset               = CircularOrbit(n_samples = 1000)

    kwargs                      = {'num_workers': c.workers, 'pin_memory': False} if cuda else {}
    train_loader                = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, **kwargs)
    valid_loader                = DataLoader(valid_dataset, batch_size=c.batch_size, shuffle=True, **kwargs)

    return train_dataset, valid_dataset, train_loader, valid_loader
