import torch


def compute_mse(pred, gt):
    """
    Compute the MSE of prediction given the gt.
    """
    return torch.nn.MSELoss()(pred, gt)


def compute_r2_total(pred, gt):
    """
    Compute the R square of prediction given the gt.
    """
    return 1 - (pred - gt).pow(2).sum() / gt.pow(2).sum()


def create_evaluator(config):
    if config.evaluator == 'mse':
        return compute_mse
    elif config.evaluator == 'r2_total':
        return compute_r2_total
    else:
        raise ValueError('Unknown evaluator: {}'.format(config.evaluator))
