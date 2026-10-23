"""Score definitions. Every score is in [0, 1] and equals 1 for two identical images."""
from __future__ import annotations

import torch

D0 = 0.1   # displacement (fraction of image size) at which the location score drops to 0.5
TAU = 1.0  # scale-score softness: score = (smaller size / larger size) ** (1 / TAU)


def location_score(dxy, d0=D0):
    """Inverse-distance score 1 / (1 + |d| / d0) of the displacement d = (dx, dy)."""
    return 1.0 / (1.0 + dxy.norm(dim=-1) / d0)


def scale_score(log_ratio, tau=TAU):
    """exp(-|log(size_B / size_A)| / tau); with tau = 1 this is smaller size / larger size."""
    return torch.exp(-log_ratio.abs() / tau)


def overall_score(pattern, location, scale, weights=(1.0, 1.0, 1.0)):
    """Weighted geometric mean, so one poor score pulls the total down."""
    w = torch.tensor(weights, dtype=pattern.dtype, device=pattern.device)
    s = torch.stack([pattern, location, scale], -1).clamp_min(1e-6)
    return torch.exp((s.log() * w).sum(-1) / w.sum())


def pair_scores(out, d0=D0, tau=TAU):
    """Scores from PatSimNet outputs (coordinates in the network's input frame)."""
    a, b = out["a"], out["b"]
    pattern = torch.sigmoid(out["pattern_logit"].float())
    return _pack(pattern, b["center"] - a["center"], b["log_size"] - a["log_size"], d0, tau)


def true_scores(batch, d0=D0, tau=TAU):
    """The same scores from a PairSynth batch's ground truth."""
    dxy = batch["center_b"] - batch["center_a"]
    return _pack(batch["pattern"], dxy, torch.log(batch["size_b"] / batch["size_a"]), d0, tau)


def _pack(pattern, dxy, log_ratio, d0, tau):
    location, scale = location_score(dxy, d0), scale_score(log_ratio, tau)
    return dict(pattern=pattern, location=location, scale=scale,
                overall=overall_score(pattern, location, scale),
                dx=dxy[..., 0], dy=dxy[..., 1], scale_ratio=log_ratio.exp())
