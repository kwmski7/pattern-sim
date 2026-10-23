"""Score definitions. Every score is in [0, 1] and equals 1 when the forecast matches the observation."""
from __future__ import annotations

import math

import torch

D0 = 0.1         # shift (fraction of the domain side) at which the location score is 0.5
ROT0_DEG = 20.0  # rotation at which the rotation score is 0.5
SCORES = ("pattern", "location", "scale", "rotation", "intensity", "overall")


def location_score(t, d0=D0):
    """1 / (1 + |shift| / d0)."""
    return 1.0 / (1.0 + t.norm(dim=-1) / d0)


def scale_score(log_s):
    """Smaller size / larger size."""
    return torch.exp(-log_s.abs())


def rotation_score(theta, rot0_deg=ROT0_DEG):
    """1 / (1 + |angle| / rot0)."""
    return 1.0 / (1.0 + theta.abs() * (180 / math.pi) / rot0_deg)


def intensity_score(log_a):
    """Smaller rain amount / larger rain amount."""
    return torch.exp(-log_a.abs())


def all_scores(params, log_amp, pattern, d0=D0, rot0_deg=ROT0_DEG):
    """params (n, 4) = (tx, ty, log_s, theta), log_amp (n,), pattern (n,) -> dict of scores; overall is
    the geometric mean of the other five."""
    s = dict(pattern=pattern, location=location_score(params[:, :2], d0), scale=scale_score(params[:, 2]),
             rotation=rotation_score(params[:, 3], rot0_deg), intensity=intensity_score(log_amp))
    s["overall"] = torch.stack(list(s.values()), -1).clamp_min(1e-6).log().mean(-1).exp()
    return s
