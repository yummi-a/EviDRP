# evidrp_v2.py -- improved EviDRP for the fair multi-omics comparison.
# Structural upgrades over MultiEviDRP (models_survey.py):
#   * channel-structured cell encoder: per-omics sub-encoders (expr/cnv/mut) with
#     softmax attention over channels -- "data organization" of the 3x128 features
#   * cell x drug interaction (Hadamard) fused with the concatenated embeddings
#   * BatchNorm + slightly wider fusion head
# Loss: annealed evidential training (pure-MSE warm-up phase, then NIG NLL +
# evidential regularizer) so the point estimate reaches MSE-level accuracy while
# the evidential head provides calibrated uncertainty.
# Input identical to survey baselines: cell [B,384] = 3 channels of 128, drug [B,128].
import torch
import torch.nn as nn
import torch.nn.functional as F

CH_DIM = 128  # per-omics channel width after PCA


def softplus1(x):
    return F.softplus(x) + 1.0


def softplus_eps(x, eps=1e-6):
    return F.softplus(x) + eps


def nig_nll(t, mu, v, a, b):
    """NIG evidential NLL (Amini et al. 2020), batch mean."""
    Omega = 2.0 * b * (1.0 + v)
    t1 = 0.5 * torch.log(torch.tensor(3.141592653589793, dtype=mu.dtype)) - 0.5 * torch.log(v)
    t2 = -a * torch.log(Omega)
    t3 = (a + 0.5) * torch.log(v * (t - mu) ** 2 + Omega)
    t4 = torch.lgamma(a) - torch.lgamma(a + 0.5)
    return (t1 + t2 + t3 + t4).mean()


def evi_reg(t, mu, v, a, lamb=0.1):
    return torch.mean(torch.abs(t - mu) * (2.0 * v + a)) * lamb


def _norm(d):
    return nn.BatchNorm1d(d)


class ChannelCellEncoder(nn.Module):
    """Per-channel MLP encoders + softmax attention over the omics channels.
    n_ch=3: expr/cnv/mut. n_ch=5: + pathway activity + tissue one-hot as
    additional biological channels (data organization via attention)."""

    def __init__(self, d_hid=128, dropout=0.1, use_bn=True, layers=1, n_ch=3,
                 attn=True):
        super().__init__()
        self.use_bn = use_bn
        self.n_ch = n_ch
        self.attn = attn
        norm = _norm if use_bn else lambda d: nn.Identity()
        encs = []
        for _ in range(n_ch):
            deps = [nn.Linear(CH_DIM, d_hid), norm(d_hid), nn.ReLU(), nn.Dropout(dropout)]
            for _ in range(layers - 1):
                deps += [nn.Linear(d_hid, d_hid), norm(d_hid), nn.ReLU(), nn.Dropout(dropout)]
            encs.append(nn.Sequential(*deps))
        self.encs = nn.ModuleList(encs)
        if attn:
            self.attn_net = nn.Sequential(nn.Linear(d_hid, d_hid), nn.ReLU(),
                                          nn.Linear(d_hid, 1))

    def forward(self, cell):
        # cell: [B, n_ch*128] -> [B, n_ch, d_hid]
        hs = torch.stack([e(cell[:, i * CH_DIM:(i + 1) * CH_DIM])
                          for i, e in enumerate(self.encs)], dim=1)
        if self.attn:
            w = torch.softmax(self.attn_net(hs).squeeze(-1), dim=1)   # [B,n_ch]
            return (hs * w.unsqueeze(-1)).sum(1), w                   # [B,d_hid]
        return hs.mean(1), None                                       # equal-weight pool


class EviDRPv2(nn.Module):
    """Channel-attention EviDRP with cell-drug interaction + NIG evidential head."""

    def __init__(self, cell_dim=384, drug_dim=128, d_hid=128, d_fus=128, dropout=0.1,
                 interact=True, use_bn=True, clu_dim=0, n_clu=16, drug_layers=1,
                 csoft_dim=0, ch_layers=1, interact_mode="hadamard",
                 gate_init=0.1, gate_fixed=False, extra_dim=0, extra_fus=False,
                 extra_chan=False, extra_chan_split=False, dt_dim=0, attn=True,
                 n_ch=3):
        super().__init__()
        self.interact = interact
        self.interact_mode = interact_mode
        self.gate_softplus = True
        self.omics_dim = n_ch * CH_DIM   # omics part of the cell vector (n_ch*128)
        self.extra_chan = extra_chan
        self.extra_chan_split = extra_chan_split
        # split mode: pathway + tissue each become their own 128-d channel
        n_ch = n_ch + (2 if extra_chan_split else (1 if extra_chan else 0))
        norm = _norm if use_bn else lambda d: nn.Identity()
        self.cell_enc = ChannelCellEncoder(d_hid, dropout, use_bn=use_bn, layers=ch_layers,
                                           n_ch=n_ch, attn=attn)
        if extra_chan:
            # project the combined extra block (pathway+tissue) to a 128-d channel
            self.extra_chan_proj = nn.Sequential(nn.Linear(extra_dim, CH_DIM),
                                                 norm(CH_DIM), nn.ReLU())
        if extra_chan_split:
            # two separate projections (pathway / tissue) as independent channels
            self.extra_chan_pw = nn.Sequential(nn.Linear(extra_dim - 28, CH_DIM),
                                               norm(CH_DIM), nn.ReLU())
            self.extra_chan_ts = nn.Sequential(nn.Linear(28, CH_DIM),
                                               norm(CH_DIM), nn.ReLU())
        if interact and interact_mode in ("gate", "gatevec"):
            g = torch.tensor(gate_init) if interact_mode == "gate" else \
                torch.full((d_hid,), gate_init)
            self.interact_gate = nn.Parameter(g)
            if gate_fixed:
                self.interact_gate.requires_grad_(False)
        self.csoft_dim = csoft_dim
        if csoft_dim > 0:
            # project soft cell-cluster membership into the cell embedding space
            self.csoft_proj = nn.Sequential(nn.Linear(csoft_dim, d_hid), nn.ReLU())
        self.extra_dim = extra_dim
        self.extra_fus = extra_fus
        if extra_dim > 0:
            # extra biological cell features (pathway activity + tissue one-hot)
            if extra_fus:
                self.extra_proj = nn.Sequential(nn.Linear(extra_dim, d_fus), nn.ReLU())
            else:
                self.extra_proj = nn.Sequential(nn.Linear(extra_dim, d_hid), nn.ReLU())
        self.clu_dim = clu_dim
        if clu_dim > 0:
            # learned embedding per drug cluster (k-means on training drugs)
            self.clu_emb = nn.Parameter(torch.randn(n_clu, clu_dim) * 0.1)
        self.dt_dim = dt_dim
        drug_in = drug_dim + clu_dim + dt_dim
        deps = [nn.Linear(drug_in, d_hid), norm(d_hid), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(drug_layers - 1):
            deps += [nn.Linear(d_hid, d_hid), norm(d_hid), nn.ReLU(), nn.Dropout(dropout)]
        self.drug_enc = nn.Sequential(*deps)
        # fusion input: [cell_v, drug_v, cell_v * drug_v] (+ interaction optional)
        fus_in = d_hid * 3 if interact else d_hid * 2
        if extra_fus:
            fus_in += d_fus
        self.fusion = nn.Sequential(nn.Linear(fus_in, d_fus), nn.ReLU(),
                                    nn.Dropout(dropout))
        # split heads: point head and evidential head share NO gradients through the
        # trunk in stage-2 calibration (only out_ev is unfrozen there)
        self.out_mu = nn.Linear(d_fus, 1)
        self.out_ev = nn.Linear(d_fus, 3)

    def forward(self, cell, drug, drug_clu=None, cell_soft=None, cell_extra=None,
            drug_dt=None):
        if self.extra_chan_split and cell_extra is not None:
            cell = torch.cat([cell[:, :self.omics_dim],
                              self.extra_chan_pw(cell_extra[:, :-28]),
                              self.extra_chan_ts(cell_extra[:, -28:])], dim=1)
        elif self.extra_chan and cell_extra is not None:
            cell = torch.cat([cell[:, :self.omics_dim],
                              self.extra_chan_proj(cell_extra)], dim=1)
        cell_v, w = self.cell_enc(
            cell[:, :self.omics_dim + (2 * CH_DIM if self.extra_chan_split
                                       else CH_DIM if self.extra_chan else 0)])
        if cell_soft is not None:
            cell_v = cell_v + self.csoft_proj(cell_soft)
        if cell_extra is not None and self.extra_fus:
            extra_v = self.extra_proj(cell_extra)
        elif cell_extra is not None:
            cell_v = cell_v + self.extra_proj(cell_extra)
        if self.clu_dim > 0 and drug_clu is not None:
            drug = torch.cat([drug, self.clu_emb[drug_clu]], dim=1)
        if self.dt_dim > 0 and drug_dt is not None:
            drug = torch.cat([drug, drug_dt], dim=1)
        drug_v = self.drug_enc(drug)
        if self.interact:
            if self.interact_mode == "norm":
                h = (cell_v * drug_v) / (cell_v.abs() * drug_v.abs() + 1e-6)
            elif self.interact_mode in ("gate", "gatevec"):
                if self.gate_softplus:
                    h = F.softplus(self.interact_gate) * cell_v * drug_v
                else:
                    h = self.interact_gate * cell_v * drug_v
            else:
                h = cell_v * drug_v
            z = torch.cat([cell_v, drug_v, h], dim=1)
        else:
            z = torch.cat([cell_v, drug_v], dim=1)
        if self.extra_fus and cell_extra is not None:
            z = torch.cat([z, extra_v], dim=1)
        z = self.fusion(z)
        mu = self.out_mu(z)[:, 0]
        o = self.out_ev(z)
        v = softplus1(o[:, 0])
        a = softplus1(o[:, 1])
        b = softplus_eps(o[:, 2])
        return mu, v, a, b


def evi2_loss(t, mu, v, a, b, ep, epochs, mse_frac=0.5, w_reg=0.1, w_mse=0.0):
    """Annealed evidential loss. During the warm-up phase train the point
    prediction with pure MSE; afterwards add the NIG NLL + regularizer. An
    optional auxiliary MSE term (w_mse>0) keeps the point estimate anchored."""
    mse = F.mse_loss(mu, t)
    if ep < mse_frac * epochs:
        return mse
    return nig_nll(t, mu, v, a, b) + evi_reg(t, mu, v, a, w_reg) + w_mse * mse
