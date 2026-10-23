"""Non-learned baselines: exhaustive registration (zoom/rotation grid + phase correlation) and
SAL-style moments (centroid shift, amount ratio, spread ratio)."""
from __future__ import annotations

import math

import numpy as np
import torch

from .data import smooth_corr, to_input
from .geometry import resample, sim_matrix

SEARCH_LOG_SCALES = np.linspace(math.log(0.7), math.log(1.4), 7)
SEARCH_ANGLES_DEG = np.arange(-40, 41, 10)


def phase_correlation(x, y, whiten=0.5):
    """Shift (n, 2) in pixels, (x, y), by which y is x moved: peak of the (partially whitened)
    cross-power spectrum of the zero-padded images x, y (n, 1, L, L)."""
    L = x.shape[-1]
    P = 2 * L
    X = torch.fft.rfft2(x[:, 0].float(), s=(P, P))
    Y = torch.fft.rfft2(y[:, 0].float(), s=(P, P))
    R = Y * X.conj()
    R = R / R.abs().clamp_min(1e-12) ** whiten
    k = torch.fft.irfft2(R, s=(P, P)).flatten(1).argmax(1)
    dy, dx = k // P, k % P
    dx = torch.where(dx >= P // 2, dx - P, dx)
    dy = torch.where(dy >= P // 2, dy - P, dy)
    return torch.stack([dx, dy], 1).float()


@torch.no_grad()
def classic_register(a, b, log_scales=SEARCH_LOG_SCALES, angles_deg=SEARCH_ANGLES_DEG):
    """Brute-force registration of forecast b onto observation a, both (n, 1, L, L) mm/h.

    For each zoom/rotation candidate: undo it, find the shift by phase correlation, and score the
    aligned pair by smoothed correlation over the overlap; keep the best candidate.
    Returns params (n, 4), log_amp (n,) and pattern (n,), like the network.
    """
    n, L = a.shape[0], a.shape[-1]
    xa, xb = to_input(a), to_input(b)
    best = torch.full((n,), -2.0, device=a.device)
    best_p = torch.zeros(n, 4, device=a.device)
    for ls in log_scales:
        for th in np.radians(angles_deg):
            zr = torch.tensor([float(ls), float(th)], device=a.device).expand(n, 2)
            lin = sim_matrix(torch.cat([torch.zeros(n, 2, device=a.device), zr], 1))
            d = phase_correlation(xa, resample(xb, lin)) * (2 / L)  # shift of the de-zoomed/de-rotated b
            t = torch.einsum("nij,nj->ni", lin[:, :2, :2], d) / 2   # the same shift in b's own frame
            p = torch.cat([t, zr], 1)
            M = sim_matrix(p)
            c = smooth_corr(a, resample(b, M), weight=resample(torch.ones_like(b), M), clamp=False)
            better = c > best
            best = torch.where(better, c, best)
            best_p = torch.where(better[:, None], p, best_p)
    M = sim_matrix(best_p)
    b_al, valid = resample(b, M), resample(torch.ones_like(b), M)
    log_amp = torch.log((b_al * valid).sum((1, 2, 3)).clamp_min(1e-6) / (a * valid).sum((1, 2, 3)).clamp_min(1e-6))
    return dict(params=best_p, log_amp=log_amp, pattern=best.clamp(0, 1))


def moments(a, b):
    """SAL-style comparison of whole fields a, b (n, 1, L, L): centroid shift of b relative to a
    (n, 2) as a fraction of the canvas (x, y), amount ratio sum(b) / sum(a), and spread ratio
    (rain-weighted radius of b / a)."""
    L = a.shape[-1]
    xs = torch.arange(L, device=a.device, dtype=torch.float32) + 0.5

    def stats(r):
        w = r[:, 0].float().clamp_min(0)
        s = w.sum((1, 2)).clamp_min(1e-6)
        wx, wy = w.sum(1), w.sum(2)  # marginals along x and y
        cx, cy = (wx * xs).sum(1) / s, (wy * xs).sum(1) / s
        var = (wx * xs ** 2).sum(1) / s - cx ** 2 + (wy * xs ** 2).sum(1) / s - cy ** 2
        return cx, cy, var.clamp_min(0).sqrt(), s

    ax, ay, a_spread, a_sum = stats(a)
    bx, by, b_spread, b_sum = stats(b)
    return dict(centroid_shift=torch.stack([bx - ax, by - ay], 1) / L, amount_ratio=b_sum / a_sum,
                spread_ratio=b_spread / a_spread.clamp_min(1e-6))
