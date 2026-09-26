import torch
from torch import nn


def mae(input, target):
    with torch.no_grad():
        return nn.L1Loss()(input, target)
