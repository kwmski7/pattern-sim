"""Metrics on a fixed, reproducible set of synthetic test pairs."""
from __future__ import annotations

import numpy as np
import torch

from .data import RELATIONS
from .scores import D0, TAU, pair_scores, true_scores

SCORES = ("pattern", "location", "scale", "overall")


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
def evaluate(model, synth, n_pairs=1024, batch_size=128, seed=1234, d0=D0, tau=TAU):
    """Same seed -> same pairs, so numbers are comparable across checkpoints."""
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=synth.device).manual_seed(seed)
    pred, true = {k: [] for k in SCORES}, {k: [] for k in SCORES}
    rel, center_err, size_err = [], [], []
    for start in range(0, n_pairs, batch_size):
        batch = synth.sample(min(batch_size, n_pairs - start), rng, g)
        out = model(batch["img_a"], batch["img_b"])
        p, t = pair_scores(out, d0, tau), true_scores(batch, d0, tau)
        for k in SCORES:
            pred[k].append(p[k].float().cpu())
            true[k].append(t[k].float().cpu())
        rel.append(batch["relation"].cpu())
        for s in "ab":
            center_err.append((out[s]["center"] - batch[f"center_{s}"]).norm(dim=1).cpu())
            size_err.append((out[s]["log_size"] - batch[f"size_{s}"].log()).abs().cpu())
    pred = {k: torch.cat(v) for k, v in pred.items()}
    true = {k: torch.cat(v) for k, v in true.items()}
    rel = torch.cat(rel)

    m = {f"{k}_mae": float((pred[k] - true[k]).abs().mean()) for k in SCORES}
    m["center_err_px"] = float(torch.cat(center_err).mean()) * synth.canvas
    m["size_err_pct"] = float(torch.cat(size_err).exp().sub(1).mean()) * 100
    m["species_auc"] = auc(pred["pattern"], rel <= 2)  # same species vs cat-dog
    m["instance_auc"] = auc(pred["pattern"][rel <= 1], rel[rel <= 1] == 0)  # same animal vs same breed
    for i, name in enumerate(RELATIONS):
        if (rel == i).any():
            m[f"pattern@{name}"] = float(pred["pattern"][rel == i].mean())
    if was_training:
        model.train()
    return m
