import torch
import wandb
from tqdm import tqdm
import os.path as osp


class Trainer:
    def __init__(self, model, loaders, optimizer, scheduler, evaluator, device, logger, config):
        self.model = model
        self.loaders = loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.device = device
        self.logger = logger
        self.config = config
        self.epoch = 0
        self.step = 0
        self.best_val_metric = -float('inf')
        self.best_test_metric = -float('inf')

    def train_epoch(self):
        self.model.train()
        train_loss, num_batches = 0, 0
        y_preds, y_trues = [], []
        for batch in tqdm(self.loaders['train']):
            x, y_true = batch
            x, y_true = x.to(self.device), y_true.to(self.device)
            self.optimizer.zero_grad()
            ret_dict = self.model(x, y_true)
            loss = ret_dict['loss']
            y_pred = ret_dict['y_pred']
            loss.backward()
            self.optimizer.step()
            self.logger.write_step({
                "step": self.step,
                "epoch": self.epoch,
                "train_loss": loss.item(),
            })
            train_loss += loss.item()
            num_batches += 1
            self.step += 1
            y_preds.append(y_pred.flatten().detach().cpu())
            y_trues.append(y_true.flatten().detach().cpu())
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)
        metric = self.evaluator(y_preds, y_trues)
        return train_loss / num_batches, metric.item()

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        val_loss, num_batches = 0, 0
        y_preds, y_trues = [], []
        for batch in tqdm(self.loaders['val']):
            x, y_true = batch
            x, y_true = x.to(self.device), y_true.to(self.device)
            ret_dict = self.model(x, y_true)
            loss = ret_dict['loss']
            y_pred = ret_dict['y_pred']
            val_loss += loss.item()
            num_batches += 1
            y_preds.append(y_pred.flatten().detach().cpu())
            y_trues.append(y_true.flatten().detach().cpu())
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)
        metric = self.evaluator(y_preds, y_trues)
        return val_loss / num_batches, metric.item()

    @torch.no_grad()
    def test_epoch(self):
        self.model.eval()
        test_loss, num_batches = 0, 0
        y_preds, y_trues = [], []
        for batch in tqdm(self.loaders['test']):
            x, y_true = batch
            x, y_true = x.to(self.device), y_true.to(self.device)
            ret_dict = self.model(x, y_true)
            loss = ret_dict['loss']
            y_pred = ret_dict['y_pred']
            test_loss += loss.item()
            num_batches += 1
            y_preds.append(y_pred.flatten().detach().cpu())
            y_trues.append(y_true.flatten().detach().cpu())
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)
        metric = self.evaluator(y_preds, y_trues)
        return test_loss / num_batches, metric.item()

    def train(self):
        for self.epoch in range(1, self.config.train.num_epochs + 1):
            train_loss, train_metric = self.train_epoch()
            val_loss, val_metric = self.val_epoch()
            test_loss, test_metric = self.test_epoch()
            self.logger.write_epoch({
                "epoch": self.epoch,
                "step": self.step,
                "train_loss": train_loss,
                "train_metric": train_metric,
                "val_loss": val_loss,
                "val_metric": val_metric,
                "test_loss": test_loss,
                "test_metric": test_metric
            })
            self.scheduler.step()
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.best_test_metric = test_metric
        self.logger.write_final({
            "best_val_metric": self.best_val_metric,
            "best_test_metric": self.best_test_metric
        })
