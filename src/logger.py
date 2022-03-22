import wandb
import os.path as osp
import os
import yaml
import json


class Logger:
    def __init__(self, config, seed):
        os.makedirs(osp.join(config.log_dir, config.title), exist_ok=True)
        with open(osp.join(config.log_dir, config.title, 'config.yml'), 'w+') as f:
            f.write(config.toYAML())
        self.log_dir = osp.join(config.log_dir, config.title, str(seed))
        os.makedirs(self.log_dir, exist_ok=True)

    def write_step(self, stat):
        wandb.log(stat)

    def write_epoch(self, stat):
        with open(osp.join(self.log_dir, 'log.json'), 'a') as f:
            f.write('{}\n'.format(stat))
        print(stat)
        wandb.log(stat)

    def write_final(self, stat):
        with open(osp.join(self.log_dir, 'final.json'), 'a') as f:
            print(file=f)
            json.dump(stat, f)
        print(stat)
        wandb.log(stat)
