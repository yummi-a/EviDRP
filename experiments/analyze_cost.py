# analyze_cost.py -- R2-M5: inference cost and model size of the uncertainty
# methods. Times forward passes on a fixed batch of the P0 test set (CPU,
# 6 threads), counts parameters, and reports per-1000-prediction wall time.
# Writes cost_table.csv.
import os
import time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

torch.set_num_threads(6)
from run_survey_seeds import make_model, load_protocol_multi, EVI2, EVI2_ARCH
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from models_survey import MultiMLP

tr, va, te, cell_idx, drug_idx, y, cells, drugs = load_protocol_multi("P0")
tc = torch.from_numpy(cells.astype(np.float32))
td = torch.from_numpy(drugs.astype(np.float32))
# full test set, chunked exactly as the real inference path
N = len(te)
c = tc[torch.from_numpy(cell_idx[te].astype(np.int64))]
d = td[torch.from_numpy(drug_idx[te].astype(np.int64))]

MC_PASSES = 50
ENS_SIZE = 5


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def timeit(fn, reps=5):
    fn()  # warmup
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        # force evaluation of every output tensor
        s = 0.0
        for o in out:
            s += float(o.float().sum().item())
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), s


rows = []
# evidential methods: single forward pass over the full test set
for name in ["evi2", "evi"]:
    m = make_model(name)
    m.eval()
    with torch.no_grad():
        if name == "evi2":
            t, _ = timeit(lambda: m(c, d, None, None, None, None))
        else:
            t, _ = timeit(lambda: m(c, d))
    rows.append({"method": name, "params": count_params(m),
                 "forward_passes": 1, "sec_full_te": t,
                 "ms_per_1000": t / N * 1000})
    print(f"{name}: {count_params(m):,} params, {t:.3f}s for {N} preds "
          f"({t/N*1000:.3f} ms/1000)")

# MC-dropout: 50 passes
m = MultiMLP()
m.eval()
with torch.no_grad():
    def mc():
        m.train()
        with torch.no_grad():
            outs = [m(c, d) for _ in range(MC_PASSES)]
        return outs
    t, _ = timeit(mc)
rows.append({"method": "mc", "params": count_params(m),
             "forward_passes": MC_PASSES, "sec_full_te": t,
             "ms_per_1000": t / N * 1000})
print(f"mc: {count_params(m):,} params x{MC_PASSES} passes, {t:.3f}s for {N} preds "
      f"({t/N*1000:.3f} ms/1000)")

# ensemble: 5 members
members = [MultiMLP() for _ in range(ENS_SIZE)]
for mm in members:
    mm.eval()
with torch.no_grad():
    def ens():
        return [mm(c, d) for mm in members]
    t, _ = timeit(ens)
rows.append({"method": "ens", "params": count_params(members[0]) * ENS_SIZE,
             "forward_passes": ENS_SIZE, "sec_full_te": t,
             "ms_per_1000": t / N * 1000})
print(f"ens: {count_params(members[0])*ENS_SIZE:,} params x{ENS_SIZE} models, "
      f"{t:.3f}s for {N} preds ({t/N*1000:.3f} ms/1000)")

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "cost_table.csv"), index=False)
print("\nwrote results/cost_table.csv")