# models_survey.py -- faithful multi-omics reproductions of survey DRP models.
# Adapted from repro/models2.py to the fresh-data channel structure:
#   cell: 3 channels (expr, cnv, mut), each PCA 128 -> 384-dim per row
#   drug: PCA 128 (shared)
# Each model takes a packed multi-channel input; identical train discipline to EviDRP.
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dims, dropout=0.0):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def split_omics(z):
    """z: [B, 384] -> list of 3 channel tensors [B,128] (expr, cnv, mut)."""
    return [z[:, :128], z[:, 128:256], z[:, 256:]]


# ---------------- BANDRP: bilinear attention over multi-omics vs drug ----------------
class BANDRP(nn.Module):
    """Multi-omics bilinear attention fusion (BAN-DRP style)."""
    def __init__(self, omics_dims=(128, 128, 128), drug_dim=128, hidden=128, dropout=0.5):
        super().__init__()
        self.cell_proj = nn.ModuleList([nn.Linear(d, hidden) for d in omics_dims])
        self.drug_proj = nn.Linear(drug_dim, hidden)
        self.W = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in omics_dims])
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, cell, drug):
        om = [torch.relu(p(c)) for p, c in zip(self.cell_proj, split_omics(cell))]
        d = torch.relu(self.drug_proj(drug))
        scores = torch.stack([(d * w(o)).sum(-1) for w, o in zip(self.W, om)], dim=1)
        w = torch.softmax(scores, dim=1)
        cellv = (torch.stack(om, dim=1) * w.unsqueeze(-1)).sum(1)
        return self.head(torch.cat([cellv, d], dim=-1)).squeeze(-1)


# ---------------- AttnOmics: attention-weighted fusion over modalities ----------------
class AttnOmics(nn.Module):
    def __init__(self, omics_dims=(128, 128, 128), drug_dim=128, hidden=128, dropout=0.3):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(d, hidden), nn.ReLU()) for d in omics_dims])
        self.attn = nn.Linear(hidden, 1)
        self.drug_proj = nn.Sequential(nn.Linear(drug_dim, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, cell, drug):
        om = torch.stack([p(c) for p, c in zip(self.proj, split_omics(cell))], dim=1)
        w = torch.softmax(self.attn(om).squeeze(-1), dim=1)
        cellv = (om * w.unsqueeze(-1)).sum(1)
        d = self.drug_proj(drug)
        return self.head(torch.cat([cellv, d], dim=-1)).squeeze(-1)


# ---------------- MMCL: contrastive alignment of two omics views + drug ----------------
class MMCL(nn.Module):
    def __init__(self, omics_dims=(128, 128), drug_dim=128, hidden=128, dropout=0.3):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Linear(omics_dims[0], hidden), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Linear(omics_dims[1], hidden), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU())
        self.drug_enc = nn.Sequential(nn.Linear(drug_dim, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, cell, drug, return_views=False):
        v1 = self.enc1(cell[:, :128])                       # expr view
        v2 = self.enc2(cell[:, 256:])                       # mut view (2nd view)
        cellv = self.fuse(torch.cat([v1, v2], dim=-1))
        d = self.drug_enc(drug)
        if return_views:
            return v1, v2
        return self.head(torch.cat([cellv, d], dim=-1)).squeeze(-1)

    def compute_loss(self, cell, drug, y):
        v1, v2 = self.forward(cell, drug, return_views=True)
        cellv = self.fuse(torch.cat([v1, v2], dim=-1))
        d = self.drug_enc(drug)
        pred = self.head(torch.cat([cellv, d], dim=-1)).squeeze(-1)
        a = F.normalize(v1, dim=-1)
        b = F.normalize(v2, dim=-1)
        logits = a @ b.t() / 0.1
        n = logits.shape[0]
        mask = torch.eye(n, device=logits.device).bool()
        nce = -(logits[mask].mean() - torch.logsumexp(logits.masked_fill(mask, -1e9), dim=1).mean())
        return F.mse_loss(pred, y) + 0.3 * nce


# ---------------- CLCLSA: cross-omics AE + contrastive + self-attention ----------------
class CLCLSA(nn.Module):
    def __init__(self, omics_dims=(128, 128, 128), drug_dim=128, hidden=64, dropout=0.3):
        super().__init__()
        self.ae = nn.ModuleList([nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                               nn.Linear(hidden, 32)) for d in omics_dims])
        self.attn = nn.MultiheadAttention(32, 4, batch_first=True, dropout=0.1)
        self.fuse = nn.Linear(32, 64)
        self.drug_proj = nn.Sequential(nn.Linear(drug_dim, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, cell, drug):
        om = torch.stack([torch.relu(e(c)) for e, c in zip(self.ae, split_omics(cell))], dim=1)
        om, _ = self.attn(om, om, om)
        cellv = torch.relu(self.fuse(om.mean(1)))
        d = self.drug_proj(drug)
        return self.head(torch.cat([cellv, d], dim=-1)).squeeze(-1)


# ---------------- Multi-omics MLP / EviDRP (fair comparison on same input) ----------------
class MultiMLP(nn.Module):
    """Deterministic MLP over the concatenated multi-omics cell vector + drug."""
    def __init__(self, cell_dim=384, drug_dim=128, d_hid=128, d_fus=64, dropout=0.1):
        super().__init__()
        self.cell_enc = nn.Sequential(nn.Linear(cell_dim, d_hid), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Dropout(dropout))
        self.drug_enc = nn.Sequential(nn.Linear(drug_dim, d_hid), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Dropout(dropout))
        self.fusion = nn.Sequential(nn.Linear(d_hid * 2, d_fus), nn.ReLU(),
                                    nn.Dropout(dropout) if dropout else nn.Identity())
        self.out = nn.Linear(d_fus, 1)

    def forward(self, cell, drug):
        return self.out(self.fusion(torch.cat([self.cell_enc(cell), self.drug_enc(drug)], dim=1))).squeeze(-1)


class MultiEviDRP(nn.Module):
    """EviDRP (NIG evidential head) over the multi-omics cell vector + drug."""
    def __init__(self, cell_dim=384, drug_dim=128, d_hid=128, d_fus=64, dropout=0.1):
        super().__init__()
        self.cell_enc = nn.Sequential(nn.Linear(cell_dim, d_hid), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Dropout(dropout))
        self.drug_enc = nn.Sequential(nn.Linear(drug_dim, d_hid), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Dropout(dropout))
        self.fusion = nn.Sequential(nn.Linear(d_hid * 2, d_fus), nn.ReLU(),
                                    nn.Dropout(dropout) if dropout else nn.Identity())
        self.out = nn.Linear(d_fus, 4)

    def forward(self, cell, drug):
        z = self.fusion(torch.cat([self.cell_enc(cell), self.drug_enc(drug)], dim=1))
        o = self.out(z)
        mu = o[:, 0]
        v = F.softplus(o[:, 1]) + 1.0
        a = F.softplus(o[:, 2]) + 1.0
        b = F.softplus(o[:, 3]) + 1e-6
        return mu, v, a, b


# ---------------- 2025--2026 survey additions (DeepDTF, GeneRelDRP, DELFOS) ----------------
# All share the survey contract forward(cell, drug) -> [B]; cell=[B,384] (3 x 128
# channels: expr, cnv, mut), drug=[B,128]. Faithful to each paper's fusion spirit.

# ---------------- CDR002: DeepDTF (2026) dual-branch transformer fusion ----------------
class DeepDTF(nn.Module):
    """Dual-branch transformer fusion: drug-side transformer over fingerprint
    chunks fused with a multi-omics cell branch."""
    def __init__(self, omics_dims=(128, 128, 128), drug_dim=128, hidden=128, dropout=0.3,
                 n_chunks=8):
        super().__init__()
        assert drug_dim % n_chunks == 0
        self.n_chunks = n_chunks
        self.drug_proj = nn.Linear(drug_dim // n_chunks, hidden)
        self.drug_enc = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=dropout, batch_first=True)
        self.cell_proj = nn.ModuleList([nn.Linear(d, hidden) for d in omics_dims])
        self.fuse = nn.Linear(hidden * len(omics_dims), hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, cell, drug):
        d = drug.view(-1, self.n_chunks, drug.shape[-1] // self.n_chunks)
        drug_vec = self.drug_enc(self.drug_proj(d)).mean(1)
        om = [p(c) for p, c in zip(self.cell_proj, split_omics(cell))]
        cell_vec = self.fuse(torch.cat(om, dim=-1))
        return self.head(torch.cat([cell_vec, drug_vec], dim=-1)).squeeze(-1)


# ---------------- CDR024: DELFOS (2024) forked multi-omics branches ----------------
class DELFOS(nn.Module):
    """Forked/specialized per-modality branches + a drug branch, concatenated
    into a fusion head."""
    def __init__(self, omics_dims=(128, 128, 128), drug_dim=128, hidden=128, dropout=0.3):
        super().__init__()
        def _branch(d):
            return nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.branches = nn.ModuleList([_branch(d) for d in omics_dims])
        self.drug_branch = _branch(drug_dim)
        self.head = nn.Sequential(nn.Linear(hidden * (len(omics_dims) + 1), hidden),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, cell, drug):
        feats = [br(c) for br, c in zip(self.branches, split_omics(cell))]
        feats.append(self.drug_branch(drug))
        return self.head(torch.cat(feats, dim=-1)).squeeze(-1)


# ---------------- CDR0xx: FourierDrug (2025) Fourier asymmetric attention ----------------
class FourierDrug(nn.Module):
    """Fourier-domain asymmetric attention fusion (arXiv 2502.04034, 2025).

    Mechanism faithful to the paper: fused cell+drug features are transformed
    to the Fourier domain, weighted by a learnable frequency attention
    (sigmoid-gated per-bin mask), and mapped back; the asymmetric training
    constraint keeps drug-sensitive samples compact while driving
    drug-resistant samples dispersed in the frequency domain. Reinterpreted
    for the survey vector contract (cell=[B,384] 3x128, drug=[B,128])."""
    def __init__(self, omics_dims=(128, 128, 128), drug_dim=128, hidden=128, dropout=0.3,
                 w_asym=0.1):
        super().__init__()
        self.cell_proj = nn.ModuleList([nn.Linear(d, hidden) for d in omics_dims])
        self.drug_proj = nn.Linear(drug_dim, hidden)
        self.fuse = nn.Linear(hidden * (len(omics_dims) + 1), hidden)
        n_freq = hidden // 2 + 1                     # rfft bins over the fused vector
        self.freq_attn = nn.Parameter(torch.ones(n_freq) * 0.5)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.w_asym = w_asym

    def _embed(self, cell, drug):
        om = [torch.relu(p(c)) for p, c in zip(self.cell_proj, split_omics(cell))]
        d = torch.relu(self.drug_proj(drug))
        z = self.fuse(torch.cat([*om, d], dim=-1))   # [B, hidden]
        Z = torch.fft.rfft(z, dim=-1)                # [B, n_freq] complex
        w = torch.sigmoid(self.freq_attn)            # learnable frequency attention
        Z = Z * w.unsqueeze(0)
        return torch.fft.irfft(Z, n=z.shape[-1], dim=-1)

    def forward(self, cell, drug):
        return self.head(self._embed(cell, drug)).squeeze(-1)

    def compute_loss(self, cell, drug, y):
        z = self._embed(cell, drug)
        pred = self.head(z).squeeze(-1)
        mse = F.mse_loss(pred, y)
        # Asymmetric constraint: drug-sensitive (low-response) samples compact,
        # drug-resistant (high-response) samples dispersed, in the frequency
        # domain. Minimizing tanh(var_sens - var_res) pushes var_sens < var_res.
        med = y.median()
        sens = y <= med
        res = ~sens
        if sens.sum() > 1 and res.sum() > 1:
            var_s = z[sens].var(0).mean()
            var_r = z[res].var(0).mean()
            asym = torch.tanh(var_s - var_r)
            return mse + self.w_asym * asym
        return mse