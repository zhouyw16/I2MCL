import numpy as np
from scipy.special import lambertw
import torch
import torch.nn as nn


class Superloss(nn.Module):
    def __init__(self, tau=0.0, lam=1.0, fac=0.9):
        super(Superloss, self).__init__()
        self.tau = tau
        self.lam = lam
        self.fac = fac


    def forward(self, loss):
        origin_loss = loss.detach().cpu().numpy()
        self.loss_mean = origin_loss.mean()
        if self.tau == 0.0: self.tau = self.loss_mean
        if self.fac > 0.0: self.tau = self.fac * self.tau + (1.0 - self.fac) * self.loss_mean

        beta = (origin_loss - self.tau) / self.lam
        gamma = -2.0 / np.exp(1.0) + 1e-12
        sigma = np.exp(-lambertw(0.5 * np.maximum(beta, gamma)).real)
        self.sigma = torch.from_numpy(sigma).to(loss.device)
        super_loss = (loss - self.tau) * self.sigma + self.lam * (torch.log(self.sigma) ** 2)
        return torch.mean(super_loss)


    def clone(self, loss):
        super_loss = (loss - self.tau) * self.sigma + self.lam * (torch.log(self.sigma) ** 2)
        return torch.mean(super_loss)


if __name__ == '__main__':
    x = torch.arange(-3, 3.05, 0.1)
    sp = Superloss()
    sp(x)
    print(sp.sigma)
