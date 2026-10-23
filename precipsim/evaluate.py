"""Metrics on a fixed, reproducible set of synthetic test pairs, for the network and the baselines."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from .classic import classic_register, moments
from .data import smooth_corr
from .scores import all_scores


def auc(score, positive):
    """Area under the ROC curve (Mann-Whitney U)."""
    positive = positive.bool()
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = torch.empty(len(score), dtype=torch.float64)
    ranks[score.argsort()] = torch.arange(1, len(score) + 1, dtype=torch.float64)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def evaluate(model, synth, n_pairs=1024, batch_size=128, seed=1234, refine=1, baseline=False):
    """Errors against the exact labels. Geometry/amount use 'same' pairs; pattern uses all pairs.

    model may be None (baselines only). With baseline=True the brute-force registration ("base_")
    and SAL-style moments ("moments_") are scored too; "rawcorr_" is the plain gridpoint
    correlation without any alignment, used as a pattern estimate.
    """
    if model is not None:
        was_training = model.training
        model.eval()
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=synth.device).manual_seed(seed)
    rows = defaultdict(list)
    for start in range(0, n_pairs, batch_size):
        batch = synth.sample(min(batch_size, n_pairs - start), rng, g)
        for k in ("params", "log_a", "pattern", "same"):
            rows[k].append(batch[k])
        rows["rawcorr_pattern"].append(smooth_corr(batch["a"], batch["b"]))
        if model is not None:
            out = model(batch["a"], batch["b"], refine=refine)
            rows["net_params"].append(out["params"])
            rows["net_log_b"].append(out["log_b"])
            rows["net_log_amp"].append(out["log_amp"])
            rows["net_pattern"].append(torch.sigmoid(out["pattern_logit"]))
        if baseline:
            c = classic_register(batch["a"], batch["b"])
            rows["base_params"].append(c["params"])
            rows["base_log_amp"].append(c["log_amp"])
            rows["base_pattern"].append(c["pattern"])
            mo = moments(batch["a"], batch["b"])
            rows["mom_shift"].append(mo["centroid_shift"])
            rows["mom_log_amp"].append(mo["amount_ratio"].log())
    cat = {k: torch.cat(v).float().cpu() for k, v in rows.items()}
    same = cat["same"].bool()
    true_p, true_a, L = cat["params"][same], cat["log_a"][same], synth.canvas
    m = {}

    def geometry(prefix, params, log_amp):
        e = params[same] - true_p
        m[f"{prefix}shift_err_px"] = float(e[:, :2].norm(dim=1).mean()) * L
        m[f"{prefix}scale_err_pct"] = float(e[:, 2].abs().exp().sub(1).mean()) * 100
        m[f"{prefix}rot_err_deg"] = float(e[:, 3].abs().mean()) * 180 / math.pi
        m[f"{prefix}amp_err_pct"] = float((log_amp[same] - true_a).abs().exp().sub(1).mean()) * 100

    def pattern(prefix, p):
        m[f"{prefix}pattern_mae"] = float((p - cat["pattern"]).abs().mean())
        m[f"{prefix}pattern_r"] = float(torch.corrcoef(torch.stack([p, cat["pattern"]]))[0, 1])
        m[f"{prefix}same_vs_other_auc"] = auc(p, same)

    if model is not None:
        geometry("", cat["net_params"], cat["net_log_amp"])
        pattern("", cat["net_pattern"])
        pred = all_scores(cat["net_params"], cat["net_log_amp"], cat["net_pattern"])
        true = all_scores(cat["params"], cat["log_a"], cat["pattern"])
        for k in ("location", "scale", "rotation", "intensity"):
            m[f"{k}_mae"] = float((pred[k][same] - true[k][same]).abs().mean())
        err = (cat["net_params"][same, :2] - true_p[:, :2]).abs().mean()
        m["unc_ratio_shift"] = float(err / cat["net_log_b"][same, :2].exp().mean())  # ~1 if calibrated
        if was_training:
            model.train()
    if baseline:
        geometry("base_", cat["base_params"], cat["base_log_amp"])
        pattern("base_", cat["base_pattern"])
        m["moments_shift_err_px"] = float((cat["mom_shift"][same] - true_p[:, :2]).norm(dim=1).mean()) * L
        m["moments_amp_err_pct"] = float((cat["mom_log_amp"][same] - true_a).abs().exp().sub(1).mean()) * 100
    pattern("rawcorr_", cat["rawcorr_pattern"])
    return m


def summary_table(m):
    """Network vs. baselines side by side (needs evaluate(..., baseline=True))."""
    import pandas as pd

    rows = {
        "shift error (grid pts)": ("shift_err_px", "base_shift_err_px", "moments_shift_err_px"),
        "size error (%)": ("scale_err_pct", "base_scale_err_pct", None),
        "rotation error (deg)": ("rot_err_deg", "base_rot_err_deg", None),
        "amount error (%)": ("amp_err_pct", "base_amp_err_pct", "moments_amp_err_pct"),
        "pattern MAE": ("pattern_mae", "base_pattern_mae", "rawcorr_pattern_mae"),
        "pattern corr. with truth": ("pattern_r", "base_pattern_r", "rawcorr_pattern_r"),
    }
    table = {name: {"network": m.get(a), "search + phase corr.": m.get(b),
                    "moments / raw corr.": m.get(c) if c else None} for name, (a, b, c) in rows.items()}
    return pd.DataFrame(table).T.astype(float).round(3)
