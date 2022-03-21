from munch import DefaultMunch
import argparse
import torch
import random
import numpy as np
import configparser
from munch import DefaultMunch


def seed_all(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config():
    """
    Loads the config file.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    config_file = args.config
    import yaml
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DefaultMunch.fromDict(config)
    return config


def create_optimizer(config, model):
    lr = config.optimizer.lr
    if config.optimizer.name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif config.optimizer.name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(
            "Unknown optimizer type {}".format(config.optimizer.name))


def create_loss_fn(config):
    if config.model.loss_fn == 'MSELoss':
        return torch.nn.MSELoss()
    elif config.model.loss_fn == 'NLLLoss':
        return torch.nn.NLLLoss()
    else:
        raise ValueError("Unknown loss type {}".format(config.model.loss_fn))


def create_scheduler(optimizer, config):
    num_epochs = config.train.num_epochs
    if config.optimizer.scheduler == 'none':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=num_epochs)
    elif config.optimizer.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=num_epochs)
    else:
        raise ValueError('Scheduler {} not supported'.format(
            config.optimizer.scheduler))
    return scheduler
