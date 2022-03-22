import torch
import torch.nn.functional as F
from utils import create_loss_fn


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout, loss_fn=None, return_dict=True):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels[0]))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels[0]))
        for i in range(1, len(hidden_channels)):
            self.lins.append(torch.nn.Linear(
                hidden_channels[i-1], hidden_channels[i]))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels[i]))
        self.lins.append(torch.nn.Linear(hidden_channels[-1], out_channels))
        self.dropout = dropout
        self.loss_fn = loss_fn
        self.return_dict = return_dict

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, y_true=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        if self.loss_fn is not None and self.return_dict and y_true is not None:
            x = x.flatten()
            return {"loss": self.loss_fn(x, y_true), "y_pred": x}
        else:
            return x

    @classmethod
    def from_config(cls, config):
        loss_fn = create_loss_fn(config)
        return cls(config.model.in_channels, config.model.hidden_channels,
                   config.model.out_channels, config.model.dropout, loss_fn)


class ConditionalAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, loss_fn):
        super(ConditionalAutoencoder, self).__init__()
        self.beta = MLP(
            in_channels, hidden_channels[:-1], hidden_channels[-1], dropout, return_dict=False)
        self.factor = torch.nn.Linear(in_channels, hidden_channels[-1])
        self.loss_fn = loss_fn

    def reset_parameters(self):
        self.beta.reset_parameters()
        self.factor.reset_parameters()

    def forward(self, xs, y_trues):
        num_batches, loss = 0, 0
        r_preds = []
        for z, r_true in zip(xs, y_trues):
            r_true.unsqueeze_(1)
            # z: (N, P), r_true: (N,)
            beta = self.beta(z)  # beta: (N, K)
            x = torch.inverse(z.T @ z) @ z.T @ r_true  # x: (K,1)
            factor = self.factor(x.view(1, -1))  # factor: (1, K)
            r_pred = beta @ factor.view(-1, 1)  # r_pred: (N, 1)
            loss += self.loss_fn(r_pred, r_true)
            r_preds.append(r_pred)
            num_batches += 1
        return {"loss": loss / num_batches, "y_preds": r_preds}

    @classmethod
    def from_config(cls, config):
        loss_fn = create_loss_fn(config)
        return cls(config.model.in_channels, config.model.hidden_channels, config.model.dropout, loss_fn)


model_dict = {
    'ConditionalAutoencoder': ConditionalAutoencoder,
}


def create_model(config):
    model = model_dict[config.model.name].from_config(config)
    return model
