# models.py -- PyTorch models for drug-response prediction with uncertainty.
# Shared two-branch architecture: cell branch + drug branch -> fusion -> head.
#   head variants:
#     deterministic : single output (MSE)
#     evidential    : NIG head (mu, nu, alpha, beta) -- deep evidential regression
# Uncertainty at inference:
#     deterministic : none (MC-dropout wrapper adds sampling)
#     evidential    : aleatoric = beta/(alpha-1); epistemic = beta/(nu*(alpha-1))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def softplus1(x):
    """softplus + 1, for nu/alpha (>=1)."""
    return F.softplus(x) + 1.0


def softplus_eps(x, eps=1e-6):
    return F.softplus(x) + eps


class Encoder(nn.Module):
    """Small MLP branch with optional dropout."""
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_hid)
        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.do(F.relu(self.fc1(x)))
        return self.do(F.relu(self.fc2(x)))


class BaseDRP(nn.Module):
    """Shared two-branch backbone producing a 64-dim fusion embedding."""
    def __init__(self, cell_dim=256, drug_dim=128, d_hid=128, d_fus=64,
                 dropout=0.1, head="deterministic"):
        super().__init__()
        self.cell_enc = Encoder(cell_dim, d_hid, dropout)
        self.drug_enc = Encoder(drug_dim, d_hid, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(d_hid * 2, d_fus), nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.head = head
        if head == "deterministic":
            self.out = nn.Linear(d_fus, 1)
        elif head == "evidential":
            self.out = nn.Linear(d_fus, 4)

    def embedding(self, cell, drug):
        return self.fusion(torch.cat([self.cell_enc(cell), self.drug_enc(drug)], dim=1))

    def forward(self, cell, drug):
        z = self.embedding(cell, drug)
        if self.head == "deterministic":
            return self.out(z).squeeze(-1)
        o = self.out(z)
        mu = o[:, 0]
        v = softplus1(o[:, 1])
        a = softplus1(o[:, 2])
        b = softplus_eps(o[:, 3])
        return mu, v, a, b


# ---------------------------------------------------------------- losses
def evidential_nll(y, mu, v, a, b, lamb=0.1):
    """NIG evidential loss (Amini et al. 2020) + evidential regularizer.
       y ~ N(mu, sigma^2); sigma^2 = b/(a-1); prior params from NIG(a,b)."""
    Omega = 2.0 * b * (1.0 + v)
    t1 = 0.5 * torch.log(torch.tensor(np.pi, dtype=mu.dtype)) - 0.5 * torch.log(v)
    t2 = -a * torch.log(Omega)
    t3 = (a + 0.5) * torch.log(v * (y - mu) ** 2 + Omega)
    t4 = torch.lgamma(a) - torch.lgamma(a + 0.5)
    nll = (t1 + t2 + t3 + t4).mean()
    reg = torch.mean(torch.abs(y - mu) * (2.0 * v + a)) * lamb
    return nll + reg


def mse_loss(y, mu):
    return F.mse_loss(mu, y)


# ---------------------------------------------------------------- inference utils
def pred_deterministic(model, cell, drug):
    with torch.no_grad():
        return model(cell, drug).numpy()


def pred_evidential(model, cell, drug):
    with torch.no_grad():
        mu, v, a, b = model(cell, drug)
    mu = mu.numpy()
    ale = (b / (a - 1.0)).numpy()            # aleatoric variance
    epi = (b / (v * (a - 1.0))).numpy()      # epistemic variance
    total = ale + epi
    return mu, np.sqrt(ale), np.sqrt(epi), np.sqrt(total)


def pred_mc_dropout(model, cell, drug, T=50):
    """MC-dropout: T stochastic forward passes. Returns mean and std."""
    model.train()  # enable dropout
    preds = np.zeros((cell.shape[0], T))
    with torch.no_grad():
        for t in range(T):
            preds[:, t] = model(cell, drug).numpy()
    model.eval()
    return preds.mean(1), preds.std(1)


def pred_ensemble(models, cell, drug):
    preds = np.zeros((cell.shape[0], len(models)))
    with torch.no_grad():
        for i, m in enumerate(models):
            preds[:, i] = m(cell, drug).numpy()
    return preds.mean(1), preds.std(1)
