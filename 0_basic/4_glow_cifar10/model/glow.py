import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
import scipy.linalg

import matplotlib.pyplot as plt


class Invertible1x1conv(nn.Module):
    def __init__(self, n_chn):
        super(Invertible1x1conv, self).__init__()
        self.n_chn = n_chn

        # sample a random orthoronal matrix to initialize weights
        self.weight = torch.qr(torch.randn(n_chn, n_chn))[0]

    def forward(self, x, logdet=None, reverse=False):
        # compute log determinant
        _, logabsdet = torch.slogdet(self.weight)
        dlogdet = x.size(-2) * x.size(-1) * logabsdet / x.size(0)

        if not reverse:
            # forward pass
            w = self.weight
            w = w.view(self.n_chn, self.n_chn, 1, 1)
            y = F.conv2d(x, w)
            #             print('conv1x1: {:.3f}'.format(dlogdet))
            if logdet is not None:
                logdet = logdet + dlogdet

            return y, logdet
        else:
            # backward pass
            w_inv = torch.inverse(self.weight)
            w_inv = w_inv.view(self.n_chn, self.n_chn, 1, 1)
            y = F.conv2d(x, w_inv)

            if logdet is not None:
                logdet = logdet - dlogdet

            return y, logdet


class ActNorm(nn.Module):
    def __init__(self, n_chn):
        super(ActNorm, self).__init__()
        self.logs = nn.Parameter(torch.zeros((1, n_chn, 1, 1), dtype=torch.float, requires_grad=True))
        self.b = nn.Parameter(torch.zeros((1, n_chn, 1, 1), dtype=torch.float, requires_grad=True))

    def forward(self, x, logdet=None, reverse=False):
        dlogdet = x.size(-2) * x.size(-1) * torch.sum(torch.abs(self.logs)) / x.size(0)
        if not reverse:
            #             print('actnorm: {:.3f}'.format(dlogdet))

            # forward pass
            y = x * torch.exp(self.logs) + self.b

            if logdet is not None:
                logdet = logdet + dlogdet

            return y, logdet
        else:
            # backward pass
            y = (x - self.b) * torch.exp(self.logs)

            if logdet is not None:
                logdet = logdet - dlogdet

            return y, logdet

class AffineNN(nn.Module):
    def __init__(self, n_chn):
        super(AffineNN, self).__init__()
        self.n_chn = n_chn
        self.n_half = n_chn // 2
        self.conv1 = nn.Conv2d(n_chn, 2 * n_chn, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2 * n_chn, 2 * n_chn, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        res = x
        x = self.conv2(x)
        x = x + res
        return x[:, :self.n_chn], x[:, self.n_chn:]


class AffineCoupling(nn.Module):
    def __init__(self, n_chn):
        super(AffineCoupling, self).__init__()
        self.n_chn = n_chn
        self.n_half = n_chn // 2
        self.transform = AffineNN(n_chn // 2)

    def forward(self, x, logdet=None, reverse=False):
        # split
        xa, xb = x[:, :self.n_half], x[:, self.n_half:]
        # affine transform
        logs, b = self.transform(xa)
        dlogdet = torch.sum(torch.abs(logs)) / x.size(0)
        # print(torch.abs(logs).size())
        if not reverse:
            #             print('affine coupling: {:.3f}'.format(dlogdet))

            # forward pass
            xb2 = xb * torch.exp(logs) + b

            # concatenate
            y = torch.cat((xa, xb2), dim=1)

            if logdet is not None:
                logdet = logdet + dlogdet

            return y, logdet
        else:
            # backward pass
            xb2 = (xb - b) * torch.exp(-logs)

            # concatenate
            y = torch.cat((xa, xb2), dim=1)

            if logdet is not None:
                logdet = logdet - dlogdet

            return y, logdet


class Squeeze2x2(nn.Module):
    def __init__(self):
        super(Squeeze2x2, self).__init__()

    def forward(self, x, reverse=False):
        n_chn, dim_hor, dim_ver = x.size()[1:]

        if not reverse:
            x = x.view(-1, n_chn * 4, dim_hor // 2, dim_ver // 2)
            return x
        else:
            x = x.view(-1, n_chn // 4, dim_hor * 2, dim_ver * 2)
            return x


class FlowStep(nn.Module):
    def __init__(self, n_chn):
        super(FlowStep, self).__init__()
        self.n_chn = n_chn
        self.actnorm = ActNorm(n_chn)
        self.inv1x1conv = Invertible1x1conv(n_chn)
        self.affcoupling = AffineCoupling(n_chn)

    def forward(self, x, logdet=None, reverse=False):
        if not reverse:
            # Forward pass
            y, logdet = self.actnorm(x, logdet=logdet, reverse=False)
            y, logdet = self.inv1x1conv(y, logdet=logdet, reverse=False)
            y, logdet = self.affcoupling(y, logdet=logdet, reverse=False)
            return y, logdet

        else:
            # Backward pass
            y, logdet = self.affcoupling(x, logdet=logdet, reverse=True)
            y, logdet = self.inv1x1conv(y, logdet=logdet, reverse=True)
            y, logdet = self.actnorm(y, logdet=logdet, reverse=True)
            return y, logdet


class Glow(nn.Module):
    def __init__(self, n_chn, n_flow, squeeze_layer=[0, 1, 2]):
        super(Glow, self).__init__()
        self.n_chn = n_chn
        self.n_flow = n_flow
        self.squeeze_layer = squeeze_layer

        self.squeeze2x2 = Squeeze2x2()
        self.layers = nn.ModuleList()
        for i_flow in range(n_flow):
            if i_flow in squeeze_layer:
                n_chn = n_chn * 4
            self.layers.append(FlowStep(n_chn))

    def forward(self, x, logdet=None, reverse=False):
        if not reverse:
            for i_flow in range(self.n_flow):
                if i_flow in self.squeeze_layer:
                    x = self.squeeze2x2(x, reverse=False)
                x, logdet = self.layers[i_flow](x, logdet=logdet, reverse=False)
            return x, logdet
        else:
            for i_flow in reversed(range(self.n_flow)):
                x, logdet = self.layers[i_flow](x, logdet=logdet, reverse=True)
                if i_flow in self.squeeze_layer:
                    x = self.squeeze2x2(x, reverse=True)
            return x, logdet
