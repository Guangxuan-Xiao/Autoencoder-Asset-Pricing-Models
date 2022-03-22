import torch
import wandb
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
        total_y_preds, total_y_trues = [], []
        for batch in self.loaders['train']:
            xs, y_trues = batch
            xs, y_trues = [x.to(self.device) for x in xs], [
                y.to(self.device) for y in y_trues]
            self.optimizer.zero_grad()
            ret_dict = self.model(xs, y_trues)
            loss = ret_dict['loss']
            y_preds = ret_dict['y_preds']
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
            total_y_preds += [y.flatten().detach().cpu() for y in y_preds]
            total_y_trues += [y.flatten().detach().cpu() for y in y_trues]
        total_y_preds = torch.cat(total_y_preds, dim=0)
        total_y_trues = torch.cat(total_y_trues, dim=0)
        metric = self.evaluator(total_y_preds, total_y_trues)
        return train_loss / num_batches, metric.item()

    @torch.no_grad()
    def eval_epoch(self, split):
        self.model.eval()
        total_loss, num_batches = 0, 0
        total_y_preds, total_y_trues = [], []
        for batch in self.loaders[split]:
            xs, y_trues = batch
            xs, y_trues = [x.to(self.device) for x in xs], [
                y.to(self.device) for y in y_trues]
            ret_dict = self.model(xs, y_trues)
            loss = ret_dict['loss']
            y_preds = ret_dict['y_preds']
            total_loss += loss.item()
            num_batches += 1
            total_y_preds += [y.flatten().detach().cpu() for y in y_preds]
            total_y_trues += [y.flatten().detach().cpu() for y in y_trues]
        total_y_preds = torch.cat(total_y_preds, dim=0)
        total_y_trues = torch.cat(total_y_trues, dim=0)
        metric = self.evaluator(total_y_preds, total_y_trues)
        return total_loss / num_batches, metric.item()

    def train(self):
        for self.epoch in range(1, self.config.train.num_epochs + 1):
            train_loss, train_metric = self.train_epoch()
            val_loss, val_metric = self.eval_epoch('val')
            test_loss, test_metric = self.eval_epoch('test')
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
