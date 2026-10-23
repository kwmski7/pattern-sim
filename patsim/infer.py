"""Score any two images with a trained checkpoint."""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import PatSimNet
from .scores import D0, TAU, location_score, overall_score, pair_scores, scale_score


def to_tensor(img):
    """Path / PIL image / numpy array / tensor -> float tensor (3, H, W) in [0, 1].

    Single-channel inputs are repeated to 3 channels. Non-uint8 numpy arrays (e.g. a physical
    2D field) are min-max normalised; tensors are assumed to be in [0, 1] already.
    """
    if isinstance(img, (str, os.PathLike)):
        img = Image.open(img)
    if isinstance(img, Image.Image):
        return torch.from_numpy(np.asarray(img.convert("RGB")).copy()).permute(2, 0, 1).float() / 255
    if isinstance(img, np.ndarray):
        a = img.astype(np.float32)
        if img.dtype == np.uint8:
            a /= 255
        else:
            lo, hi = np.nanmin(a), np.nanmax(a)
            a = np.nan_to_num((a - lo) / max(float(hi - lo), 1e-12))
        t = torch.from_numpy(a)
        t = t[None] if t.ndim == 2 else t.permute(2, 0, 1)
    else:
        t = torch.as_tensor(img).detach().float().cpu()
        t = t[None] if t.ndim == 2 else t
    return (t.expand(3, -1, -1) if t.shape[0] == 1 else t[:3]).clamp(0, 1)


class PairScorer:
    """scorer.score(img_a, img_b) -> dict with pattern / location / scale / overall scores.

    Images of any size or aspect are letterboxed to the training canvas. Reported centres are
    fractions of each original image's width / height (y down); sizes are the object's longest
    side as a fraction of the image's longest side; dx, dy, scale_ratio describe B relative to A.
    """

    def __init__(self, model, canvas=160, d0=D0, tau=TAU, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.canvas, self.d0, self.tau = canvas, d0, tau

    @classmethod
    def from_checkpoint(cls, path, device=None, **kwargs):
        ck = torch.load(path, map_location="cpu")
        model = PatSimNet(n_classes=len(ck["classes"]), pretrained=False)
        model.load_state_dict(ck["model"])
        return cls(model, canvas=ck["config"]["canvas"], device=device, **kwargs)

    def _letterbox(self, img):
        t, S = to_tensor(img), self.canvas
        h, w = t.shape[-2:]
        nh, nw = max(1, round(h * S / max(h, w))), max(1, round(w * S / max(h, w)))
        if (nh, nw) != (h, w):
            t = F.interpolate(t[None], size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)[0]
        y0, x0 = (S - nh) // 2, (S - nw) // 2
        canvas = torch.full((3, S, S), 0.5)
        canvas[:, y0:y0 + nh, x0:x0 + nw] = t
        return canvas.clamp(0, 1), (x0, y0, nw, nh)

    @torch.no_grad()
    def score(self, img_a, img_b, return_maps=False):
        (canvas_a, box_a), (canvas_b, box_b) = self._letterbox(img_a), self._letterbox(img_b)
        out = self.model(canvas_a[None].to(self.device), canvas_b[None].to(self.device))
        S = self.canvas
        objs = {}
        for s, (x0, y0, nw, nh) in (("a", box_a), ("b", box_b)):
            cx, cy = out[s]["center"][0].tolist()
            objs[s] = {"center": ((cx * S - x0) / nw, (cy * S - y0) / nh),
                       "size": float(out[s]["log_size"][0].exp())}
        dxy = torch.tensor(objs["b"]["center"]) - torch.tensor(objs["a"]["center"])
        log_ratio = torch.tensor(math.log(objs["b"]["size"] / objs["a"]["size"]))
        pattern = torch.sigmoid(out["pattern_logit"][0].float()).cpu()
        location, scale = location_score(dxy, self.d0), scale_score(log_ratio, self.tau)
        res = dict(pattern=float(pattern), location=float(location), scale=float(scale),
                   overall=float(overall_score(pattern, location, scale)),
                   dx=float(dxy[0]), dy=float(dxy[1]), scale_ratio=float(log_ratio.exp()),
                   a=objs["a"], b=objs["b"])
        if return_maps:
            up = lambda m: F.interpolate(m, size=(S, S), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            res["maps"] = {}
            for s, canvas in (("a", canvas_a), ("b", canvas_b)):
                o = out[s]
                heat = up(o["heat"][:1].float().flatten(1).softmax(1).view_as(o["heat"][:1]))
                res["maps"][s] = dict(canvas=canvas.permute(1, 2, 0).numpy(),
                                      center=tuple(o["center"][0].tolist()),
                                      size=float(o["log_size"][0].exp()),
                                      mask=up(torch.sigmoid(o["mask_logit"][:1].float())),
                                      heat=heat / heat.max())
        return res

    @torch.no_grad()
    def score_canvases(self, a, b):
        """Fast path for batches already at canvas size: (n, 3, S, S) tensors in [0, 1] -> dict of tensors."""
        return pair_scores(self.model(a.to(self.device), b.to(self.device)), self.d0, self.tau)
