import torch


def L2_loss(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return torch.nn.MSELoss()(x, y)

def L1_loss(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return torch.nn.L1Loss()(x, y)