"""Matplotlib helpers for rain fields, synthetic training pairs and scorer results."""
from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def _show(ax, r, title):
    cmap = matplotlib.colormaps["turbo"].copy()
    cmap.set_under("white")
    cmap.set_bad("white")
    ax.imshow(np.asarray(r, dtype=np.float32), cmap=cmap, norm=LogNorm(vmin=0.1, vmax=20), interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def show_pairs(batch, n=6):
    """Synthetic training pairs: observation (top), forecast (bottom) with the true labels.
    Shifts are (right, up) on screen in grid points; rotation counter-clockwise on screen."""
    n = min(n, batch["a"].shape[0])
    L = batch["a"].shape[-1]
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.4), squeeze=False)
    for i in range(n):
        tx, ty, ls, th = batch["params"][i].tolist()
        kind = "same" if bool(batch["same"][i]) else "other"
        _show(axes[0, i], batch["a"][i, 0].float().cpu(), f"obs   [{kind}]  pattern {float(batch['pattern'][i]):.2f}")
        _show(axes[1, i], batch["b"][i, 0].float().cpu(),
              f"shift ({tx * L:+.0f}, {-ty * L:+.0f})  size x{math.exp(ls):.2f}\n"
              f"rot {-math.degrees(th):+.0f} deg  amount x{math.exp(float(batch['log_a'][i])):.2f}")
    fig.tight_layout()
    return fig


def show_result(res, title=None):
    """Observation, forecast, and forecast with its shift/size/rotation undone, each with the observed
    0.5 and 2 mm/h contours on top; needs score(..., return_maps=True)."""
    mp = res["maps"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    _show(axes[0], mp["obs"], "observation (A)")
    _show(axes[1], mp["fcst"], "forecast (B) + observed contours")
    _show(axes[2], mp["fcst_aligned"], "forecast moved back onto A + observed contours")
    if np.nanmax(mp["obs"]) > 0.5:
        for ax in axes[1:]:
            ax.contour(mp["obs"], levels=[0.5, 2.0], colors="k", linewidths=0.6)
    km = f" = {res['shift_km']:.0f} km" if "shift_km" in res else ""
    head = f"{title}\n" if title else ""
    fig.suptitle(
        f"{head}pattern {res['pattern']:.2f} | location {res['location']:.2f} | scale {res['scale']:.2f} | "
        f"rotation {res['rotation']:.2f} | intensity {res['intensity']:.2f} | overall {res['overall']:.2f}\n"
        f"shift east {res['shift_east']:+.1f}, north {res['shift_north']:+.1f} grid pts{km} | "
        f"size x{res['scale_ratio']:.2f} | rotation {res['rotation_deg']:+.0f} deg ccw | "
        f"amount x{res['intensity_ratio']:.2f} | raw corr {res['corr_raw']:.2f}", fontsize=9)
    fig.tight_layout()
    return fig
