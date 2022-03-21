import wandb
import torch
from utils import load_config, seed_all, create_optimizer, create_scheduler
from model import create_model
from evaluator import create_evaluator
from data import create_loaders
from trainer import Trainer
from logger import Logger

if __name__ == "__main__":
    config = load_config()
    print(config)
    for seed in config.seeds:
        run = wandb.init(project="Autoencoder Asset Pricing Models",
                         config=config, name=config.title, reinit=True)
        seed_all(seed)
        logger = Logger(config, seed)
        model = create_model(config)
        device = torch.device(config.model.device)
        model.to(device)
        loaders = create_loaders(config)
        optimizer = create_optimizer(config, model)
        scheduler = create_scheduler(optimizer, config)
        evaluator = create_evaluator(config)
        trainer = Trainer(model, loaders, optimizer, scheduler,
                          evaluator, device, logger, config)
        trainer.train()
        run.finish()
