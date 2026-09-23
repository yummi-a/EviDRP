# train_utils.py -- data loading, training loop, metrics, conformal calibration.
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import r2_score, roc_auc_score
from models import evidential_nll, mse_loss, softplus1, softplus_eps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPL = os.path.join(ROOT, "splits")
FEAT = os.path.join(ROOT, "feat")


def load_protocol(P):
    """Return (tr_idx, va_idx, te_idx, cell_idx, drug_idx, y, cells, drugs)."""
    tr = np.load(os.path.join(SPL, f"{P}_train.npy"))
    va = np.load(os.path.join(SPL, f"{P}_val.npy"))
    te = np.load(os.path.join(SPL, f"{P}_test.npy"))
    cell_idx = np.load(os.path.join(SPL, "cell_idx.npy"))
    drug_idx = np.load(os.path.join(SPL, "drug_idx.npy"))
    y = np.load(os.path.join(SPL, "y.npy"))
    cells = np.load(os.path.join(FEAT, f"{P}_cells.npy"))
    drugs = np.load(os.path.join(FEAT, f"{P}_drugs.npy"))
    return tr, va, te, cell_idx, drug_idx, y, cells, drugs


def load_crossdataset(src, tgt):
    """P3: train on 80% of src dataset rows, val on 20% of src, test on all tgt rows.
       Uses P0 cell/drug PCA (same feature space across protocols)."""
    r = pd.read_csv(os.path.join(ROOT, "processed", "response_long.csv"))
    tr0, va0, te0, cell_idx, drug_idx, y, cells, drugs = load_protocol("P0")
    src_rows = r.index[r["dataset"] == src].to_numpy()
    tgt_rows = r.index[r["dataset"] == tgt].to_numpy()
    rng = np.random.RandomState(42)
    nv = int(len(src_rows) * 0.2)
    perm = rng.permutation(src_rows)
    va = np.sort(perm[:nv])
    tr = np.sort(perm[nv:])
    return tr, va, tgt_rows, cell_idx, drug_idx, y, cells, drugs


def make_batches(n, batch, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return [idx[i:i + batch] for i in range(0, n, batch)]


def train_model(model, cell_idx, drug_idx, y, cells, drugs, tr_idx, val_idx=None,
                epochs=60, batch=1024, lr=1e-3, patience=8, seed=0,
                evidential=False, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    if val_idx is None:
        # fallback: carve internal validation from train rows
        nv = max(1, int(len(tr_idx) * 0.05))
        val_rows = rng.choice(tr_idx, nv, replace=False)
        val_set = set(val_rows.tolist())
        fit_rows = np.array([r for r in tr_idx if r not in val_set])
    else:
        fit_rows = tr_idx
        val_rows = val_idx
    fit_cell = torch.from_numpy(cell_idx[fit_rows].astype(np.int64))
    fit_drug = torch.from_numpy(drug_idx[fit_rows].astype(np.int64))
    fit_y = torch.from_numpy(y[fit_rows].astype(np.float32))
    val_cell = torch.from_numpy(cell_idx[val_rows].astype(np.int64))
    val_drug = torch.from_numpy(drug_idx[val_rows].astype(np.int64))
    val_y = torch.from_numpy(y[val_rows].astype(np.float32))
    tc = torch.from_numpy(cells.astype(np.float32))
    td = torch.from_numpy(drugs.astype(np.float32))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best_rmse, best_state, bad = float("inf"), None, 0

    def loss_fn(z, t):
        if evidential:
            o = model.out(z)
            mu = o[:, 0]
            v = softplus1(o[:, 1])
            a = softplus1(o[:, 2])
            bb = softplus_eps(o[:, 3])
            return evidential_nll(t, mu, v, a, bb)
        return mse_loss(t, model.out(z).squeeze(-1))

    for ep in range(epochs):
        model.train()
        # A: cache encoder outputs (detached) -> train fusion head per batch
        with torch.no_grad():
            Cb = model.cell_enc(tc).detach()
            Db = model.drug_enc(td).detach()
        for b in make_batches(len(fit_rows), batch, seed + ep):
            z = model.fusion(torch.cat([Cb[fit_cell[b]], Db[fit_drug[b]]], dim=1))
            loss = loss_fn(z, fit_y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        # B: update encoders on full fit data once per 2 epochs (fresh graph)
        if ep % 2 == 0:
            z = model.fusion(torch.cat([model.cell_enc(tc)[fit_cell],
                                        model.drug_enc(td)[fit_drug]], dim=1))
            loss = loss_fn(z, fit_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # validate
        model.eval()
        with torch.no_grad():
            if evidential:
                mu, *_ = model(tc[val_cell], td[val_drug])
            else:
                mu = model(tc[val_cell], td[val_drug])
            rmse = float(torch.sqrt(F.mse_loss(mu, val_y)).item())
        if rmse < best_rmse - 1e-5:
            best_rmse, bad = rmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop @ep{ep} best_rmse={best_rmse:.4f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------- metrics
def rmse(y, mu):
    return float(np.sqrt(np.mean((y - mu) ** 2)))


def pearson(y, mu):
    return float(stats.pearsonr(y, mu)[0])


def spearman(a, b):
    return float(stats.spearmanr(a, b)[0])


def r2(y, mu):
    return float(r2_score(y, mu))


def conformal(y_cal, mu_cal, sig_cal, y_te, mu_te, sig_te, alpha=0.1,
              studentized=True, return_bounds=False):
    """Split-conformal prediction intervals.
       Returns (PICP, MPIW) for nominal 1-alpha coverage; with return_bounds=True
       also returns (lo, hi) arrays on the test set (for Winkler interval score).
       If studentized: score=|resid|/sigma ; else score=|resid|."""
    if studentized:
        s_cal = np.abs(y_cal - mu_cal) / (sig_cal + 1e-8)
    else:
        s_cal = np.abs(y_cal - mu_cal)
    n = len(s_cal)
    q = np.quantile(s_cal, (1 - alpha) * (1 + 1.0 / n))
    if studentized:
        lo = mu_te - q * sig_te
        hi = mu_te + q * sig_te
    else:
        lo = mu_te - q
        hi = mu_te + q
    cov = float(np.mean((y_te >= lo) & (y_te <= hi)))
    wid = float(np.mean(hi - lo))
    if return_bounds:
        return cov, wid, lo, hi
    return cov, wid


def winkler_score(y, lo, hi, alpha=0.1):
    """Winkler interval score (Gneiting & Raftery 2007) for a (1-alpha) interval
       [lo, hi]. Penalises width, and adds a penalty proportional to the
       shortfall/excess when the observation falls outside. Lower is better."""
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if lo.ndim == 0:
        lo = np.full_like(y, float(lo))
    if hi.ndim == 0:
        hi = np.full_like(y, float(hi))
    d = hi - lo
    out = np.zeros_like(y, dtype=float)
    inside = (y >= lo) & (y <= hi)
    out[inside] = d[inside]
    out[y < lo] = d[y < lo] + 2.0 / alpha * (lo[y < lo] - y[y < lo])
    out[y > hi] = d[y > hi] + 2.0 / alpha * (y[y > hi] - hi[y > hi])
    return float(np.mean(out))


def crps_mc(y, samples, n_pairs=20000):
    """Monte-Carlo CRPS from predictive samples (robust for any predictive shape,
       incl. NIG Student-t, MC-dropout mixtures and ensembles):
       CRPS = E|X-y| - 0.5 E|X-X'|. Lower is better."""
    y = np.asarray(y, dtype=float)
    samples = np.asarray(samples, dtype=float)          # [N, M] or [M]
    if samples.ndim == 1:
        samples = samples[None, :]
    n = samples.shape[0]
    rng = np.random.RandomState(0)
    # E|X-y| per observation
    e_abs = np.mean(np.abs(samples - y[None, :]), axis=0)
    # E|X-X'| via paired draws (unbiased for the expectation)
    i = rng.randint(0, n, n_pairs)
    j = rng.randint(0, n, n_pairs)
    e_pair = np.mean(np.abs(samples[i] - samples[j]), axis=0)
    return float(np.mean(e_abs - 0.5 * e_pair))


def nig_predictive_samples(mu, v, a, b, n=200, seed=0):
    """Draw n samples from the NIG predictive: y ~ N(mu, 1/lambda),
       lambda ~ Gamma(shape=a, rate=b), i.e. a Student-t. Returns [n, M]."""
    rng = np.random.RandomState(seed)
    lam = rng.gamma(shape=a, scale=1.0 / b)             # [M]
    sd = 1.0 / np.sqrt(v * lam)                         # [M]
    return mu[None, :] + sd[None, :] * rng.randn(n, mu.shape[0])


def risk_coverage(y, mu, sigma, npts=20):
    """Sort by sigma ascending; return coverage grid and RMSE at each coverage."""
    order = np.argsort(sigma)
    yy, mm = y[order], mu[order]
    grids = np.linspace(0.05, 1.0, npts)
    out = []
    for g in grids:
        k = max(1, int(g * len(yy)))
        out.append((g, rmse(yy[:k], mm[:k])))
    return np.array(out)


def ood_auc(ood_mask, sigma):
    """AUC of treating high uncertainty as OOD flag."""
    if len(np.unique(ood_mask)) < 2:
        return float("nan")
    return float(roc_auc_score(ood_mask.astype(int), sigma))
