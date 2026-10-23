"""Matplotlib helpers to look at synthetic batches and at scorer results."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .data import RELATIONS
from .scores import true_scores


def _img(t):
    return t.detach().float().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def _box(ax, center, size, S, color):
    """Square of side `size` (fraction of S) around `center` (fractions), plus a centre cross."""
    cx, cy, half = center[0] * S, center[1] * S, size * S / 2
    ax.add_patch(Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False, ec=color, lw=1.8))
    ax.plot(cx, cy, "+", color=color, ms=10, mew=2)


def show_batch(batch, n=6, pred=None):
    """n pairs (A on top, B below) with true boxes (green) and, given model outputs, predicted ones (red)."""
    n = min(n, batch["img_a"].shape[0])
    S = batch["img_a"].shape[-1]
    t = true_scores(batch)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5), squeeze=False)
    for i in range(n):
        for row, s in enumerate("ab"):
            ax = axes[row, i]
            ax.imshow(_img(batch[f"img_{s}"][i]))
            _box(ax, batch[f"center_{s}"][i].tolist(), float(batch[f"size_{s}"][i]), S, "lime")
            if pred is not None:
                _box(ax, pred[s]["center"][i].tolist(), float(pred[s]["log_size"][i].exp()), S, "red")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, i].set_title(f"{RELATIONS[int(batch['relation'][i])]}\n"
                             f"P {float(t['pattern'][i]):.2f}  L {float(t['location'][i]):.2f}  "
                             f"S {float(t['scale'][i]):.2f}", fontsize=8)
    fig.tight_layout()
    return fig


def show_result(res, title=None, show_mask=True):
    """The two letterboxed inputs with predicted object boxes; needs score(..., return_maps=True)."""
    maps = res["maps"]
    S = maps["a"]["canvas"].shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(7, 4.2))
    for ax, s, color in zip(axes, "ab", ("orange", "deepskyblue")):
        ax.imshow(maps[s]["canvas"])
        if show_mask:
            ax.imshow(maps[s]["mask"], cmap="magma", alpha=0.35, vmin=0, vmax=1)
        _box(ax, maps[s]["center"], maps[s]["size"], S, color)
        ax.set_title(f"image {s.upper()}", fontsize=9)
        ax.axis("off")
    head = f"{title}\n" if title else ""
    fig.suptitle(f"{head}pattern {res['pattern']:.2f} | location {res['location']:.2f} | "
                 f"scale {res['scale']:.2f} | overall {res['overall']:.2f}\n"
                 f"dx {res['dx']:+.2f}  dy {res['dy']:+.2f}  size B/A {res['scale_ratio']:.2f}", fontsize=10)
    fig.tight_layout()
    return fig
