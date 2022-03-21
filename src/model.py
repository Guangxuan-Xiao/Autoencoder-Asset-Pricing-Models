import torch  # Essential!
import torch.nn.functional as F
from utils import create_loss_fn


class Linear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, loss_fn=None, return_dict=True):
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.loss_fn = loss_fn
        self.return_dict = return_dict

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x, y_true=None):
        x = self.linear(x)
        if self.loss_fn is not None and self.return_dict and y_true is not None:
            x = x.flatten()
            return {"loss": self.loss_fn(x, y_true), "y_pred": x}
        else:
            return x

    @classmethod
    def from_config(cls, config):
        loss_fn = create_loss_fn(config)
        return cls(config.model.in_channels, config.model.out_channels, loss_fn)


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
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, loss_fn):
        super(ConditionalAutoencoder, self).__init__()
        self.beta = MLP(
            in_channels, hidden_channels[:-1], hidden_channels[-1], dropout, return_dict=False)
        self.factor = torch.nn.Linear(out_channels, hidden_channels[-1])
        self.loss_fn = loss_fn

    def reset_parameters(self):
        self.beta.reset_parameters()
        self.factor.reset_parameters()

    def forward(self, x, y_true):
        B, N, P = x.shape
        beta = self.beta(x.view(-1, P))
        factor = self.factor(y_true)
        beta = beta.view(B, N, -1)
        y_pred = torch.bmm(beta, factor.unsqueeze(2)).squeeze(2)
        return {"loss": self.loss_fn(y_pred, y_true), "y_pred": y_pred}

    @classmethod
    def from_config(cls, config):
        loss_fn = create_loss_fn(config)
        return cls(config.model.in_channels, config.model.hidden_channels, config.model.out_channels, config.model.dropout, loss_fn)


model_dict = {
    'MLP': MLP,
    'Linear': Linear,
    'ConditionalAutoencoder': ConditionalAutoencoder,
}


def create_model(config):
    model = model_dict[config.model.name].from_config(config)
    return model
