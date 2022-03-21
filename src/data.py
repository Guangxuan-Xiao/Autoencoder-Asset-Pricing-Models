import torch
import numpy as np
import os.path as osp


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split='train'):
        super().__init__()
        self.features = np.load(
            osp.join(path, '{}_features.npy'.format(split))).astype(np.float32)
        self.fret = np.load(
            osp.join(path, '{}_fret.npy'.format(split))).astype(np.float32)

    def __len__(self):
        return self.fret.shape[1]

    def __getitem__(self, index):
        return self.features[:, index, :], self.fret[:, index]


def create_loader(config, split):
    dataset = Dataset(config.data.path, split)
    shuffle = split == 'train'
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.train.batch_size,
                                         shuffle=shuffle, num_workers=config.data.num_workers, drop_last=config.train.drop_last)
    return loader


def create_loaders(config):
    loaders = {}
    for split in ['train', 'val', 'test']:
        loaders[split] = create_loader(config, split)
    return loaders
