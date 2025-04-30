import torch


def L2_loss(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return torch.nn.MSELoss()