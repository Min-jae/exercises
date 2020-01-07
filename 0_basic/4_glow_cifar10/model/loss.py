import torch
import torch.nn as nn

class GlowLoss(nn.Module):
    def __init__(self):
        super(GlowLoss, self).__init__()

    def forward(self, z, logdet):
        log_prior = -torch.sum(z ** 2) / 2
        nll = -(log_prior + logdet)
        return nll