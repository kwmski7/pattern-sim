"""Score a forecast field against an observed field with a trained checkpoint."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .classic import moments
from .data import smooth_corr
from .model import RegNet
from .scores import D0, ROT0_DEG, all_scores


class FieldScorer:
    """scorer.score(obs, fcst) -> scores in [0, 1] (pattern, location, scale, rotation, intensity,
    overall) plus the estimated shift, zoom, rotation and amount ratio of the forecast.

    Fields are 2D arrays on the same grid, rain in mm per `accum_hours`, row 0 = north (otherwise pass
    north_up=False). Any grid size works: fields are letterboxed onto the training canvas.
    Shifts are in grid points (and km with grid_km); rotation_deg is counter-clockwise on the map.
    """

    def __init__(self, model, canvas=128, d0=D0, rot0_deg=ROT0_DEG, refine=1, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.canvas, self.d0, self.rot0_deg, self.refine = canvas, d0, rot0_deg, refine

    @classmethod
    def from_checkpoint(cls, path, device=None, **kwargs):
        ck = torch.load(path, map_location="cpu")
        canvas = ck["config"]["canvas"]
        model = RegNet(canvas=canvas, pretrained=False)
        model.load_state_dict(ck["model"])
        return cls(model, canvas=canvas, device=device, **kwargs)

    def _prep(self, x, accum_hours):
        """(H, W) or (N, H, W) array/tensor -> letterboxed (N, 1, L, L) mm/h tensor, original (H, W)."""
        if torch.is_tensor(x):
            x = x.detach().float().cpu()
        else:
            x = torch.from_numpy(np.ascontiguousarray(np.asarray(x, dtype=np.float32)))
        if x.ndim == 2:
            x = x[None]
        x = torch.nan_to_num(x).clamp_min(0) / accum_hours
        n, h, w = x.shape
        L, side = self.canvas, max(h, w)
        nh, nw = max(1, round(h * L / side)), max(1, round(w * L / side))
        if (nh, nw) != (h, w):
            x = F.interpolate(x[:, None], size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)[:, 0]
        out = torch.zeros(n, 1, L, L)
        y0, x0 = (L - nh) // 2, (L - nw) // 2
        out[:, 0, y0:y0 + nh, x0:x0 + nw] = x
        return out, (h, w)

    @torch.no_grad()
    def score_batch(self, obs, fcst, accum_hours=1.0, grid_km=None, north_up=True, return_maps=False, batch_size=64):
        """obs, fcst: (N, H, W) or (H, W). Returns a dict of numpy arrays of length N."""
        a, shape_a = self._prep(obs, accum_hours)
        b, shape_b = self._prep(fcst, accum_hours)
        if shape_a != shape_b or a.shape[0] != b.shape[0]:
            raise ValueError(f"obs and fcst must be on the same grid: {tuple(a.shape)} {shape_a} vs {tuple(b.shape)} {shape_b}")
        side = max(shape_a)  # canvas fraction -> grid points
        keys = ("params", "log_b", "pattern_logit", "log_amp", "b_aligned")
        parts = {k: [] for k in keys}
        for s in range(0, a.shape[0], batch_size):
            out = self.model(a[s:s + batch_size].to(self.device), b[s:s + batch_size].to(self.device), refine=self.refine)
            for k in keys:
                parts[k].append(out[k].float().cpu())
        o = {k: torch.cat(v) for k, v in parts.items()}
        p, sgn = o["params"], (-1.0 if north_up else 1.0)
        unc, mo = o["log_b"].exp(), moments(a, b)
        res = all_scores(p, o["log_amp"], torch.sigmoid(o["pattern_logit"]), self.d0, self.rot0_deg)
        res.update(
            shift_east=p[:, 0] * side, shift_north=sgn * p[:, 1] * side,
            scale_ratio=p[:, 2].exp(), rotation_deg=sgn * torch.rad2deg(p[:, 3]),
            intensity_ratio=o["log_amp"].exp(),
            unc_shift=unc[:, :2].mean(1) * side, unc_scale_pct=unc[:, 2] * 100,
            unc_rotation_deg=torch.rad2deg(unc[:, 3]),
            corr_raw=smooth_corr(a, b),  # plain gridpoint correlation, no alignment (double penalty)
            centroid_east=mo["centroid_shift"][:, 0] * side, centroid_north=sgn * mo["centroid_shift"][:, 1] * side,
            amount_ratio=mo["amount_ratio"],
        )
        if grid_km is not None:
            res["shift_km"] = torch.hypot(res["shift_east"], res["shift_north"]) * grid_km
            res["unc_shift_km"] = res["unc_shift"] * grid_km
        res = {k: v.numpy() for k, v in res.items()}
        if return_maps:
            res["maps"] = dict(obs=a[:, 0].numpy(), fcst=b[:, 0].numpy(), fcst_aligned=o["b_aligned"][:, 0].numpy())
        return res

    def score(self, obs, fcst, **kwargs):
        """One pair -> dict of floats (and 2D maps with return_maps=True)."""
        res = self.score_batch(obs, fcst, **kwargs)
        maps = res.pop("maps", None)
        out = {k: float(v[0]) for k, v in res.items()}
        if maps is not None:
            out["maps"] = {k: v[0] for k, v in maps.items()}
        return out
