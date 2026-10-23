"""Precipitation fields (ERA5 or your own) and the self-supervised forecast/observation pair synthesizer."""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .geometry import sim_matrix

# Google's public analysis-ready ERA5 (hourly, 0.25 deg), readable without a login.
ARCO_ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def to_input(r):
    """Rain rate (mm/h) -> network input log1p(r / 0.1 mm/h) / 5, roughly 0..1.2."""
    return torch.log1p(r.clamp_min(0) / 0.1) / 5


def gaussian_blur(x, sigma, k=9):
    """Per-sample Gaussian blur of x (n, 1, H, W) with sigma (n,) in pixels; sigma 0 leaves x unchanged."""
    n, h, w = x.shape[0], x.shape[-2], x.shape[-1]
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    kern = torch.exp(-ax ** 2 / (2 * sigma.to(x.dtype).clamp_min(1e-3).view(n, 1) ** 2))
    kern = kern / kern.sum(1, keepdim=True)
    x = F.conv2d(x.reshape(1, n, h, w), kern.view(n, 1, 1, k), padding=(0, k // 2), groups=n)
    x = F.conv2d(x, kern.view(n, 1, k, 1), padding=(k // 2, 0), groups=n)
    return x.view(n, 1, h, w)


def smooth_corr(a, b, sigma=1.5, weight=None, clamp=True):
    """Correlation of smoothed log rain (to_input) per sample; optional pixel weights (n, 1, H, W)."""
    n = a.shape[0]
    s = torch.full((n,), sigma, device=a.device)
    x, y = gaussian_blur(to_input(a), s).flatten(1), gaussian_blur(to_input(b), s).flatten(1)
    w = torch.ones_like(x) if weight is None else weight.flatten(1)
    sw = w.sum(1, keepdim=True).clamp_min(1e-6)
    x = x - (w * x).sum(1, keepdim=True) / sw
    y = y - (w * y).sum(1, keepdim=True) / sw
    vxy = (w * x * x).sum(1) * (w * y * y).sum(1)
    corr = torch.where(vxy > 1e-8, (w * x * y).sum(1) / vxy.clamp_min(1e-8).sqrt(), torch.zeros_like(vxy))
    return corr.clamp(0, 1) if clamp else corr


@dataclass
class FieldBank:
    """Precipitation fields on one grid, sorted by time."""

    precip: torch.Tensor  # float16 (N, H, W) rain rate in mm/h, row 0 = northernmost
    time: np.ndarray      # datetime64[s] (N,)
    lat: np.ndarray       # (H,) degrees, descending
    lon: np.ndarray       # (W,) degrees

    def __len__(self):
        return self.precip.shape[0]

    def to(self, device):
        return FieldBank(self.precip.to(device), self.time, self.lat, self.lon)

    def select_years(self, first, last):
        years = self.time.astype("datetime64[Y]").astype(int) + 1970
        keep = np.flatnonzero((years >= first) & (years <= last))
        return FieldBank(self.precip[torch.as_tensor(keep, device=self.precip.device)], self.time[keep],
                         self.lat, self.lon)

    def index_of(self, when):
        """Index of the field closest in time to `when`, e.g. '2023-08-09T12'."""
        return int(np.argmin(np.abs(self.time - np.datetime64(when, "s"))))

    def px(self, lat, lon):
        """Pixel coordinates (x, y) of a lat/lon point; pixel k spans [k, k + 1)."""
        return ((lon - self.lon[0]) / (self.lon[1] - self.lon[0]) + 0.5,
                (self.lat[0] - lat) / (self.lat[0] - self.lat[1]) + 0.5)

    def crop(self, i, lat, lon, size=128):
        """size x size numpy crop (mm/h) of field i centred near (lat, lon)."""
        x, y = self.px(lat, lon)
        x0 = int(np.clip(round(x - size / 2), 0, self.precip.shape[2] - size))
        y0 = int(np.clip(round(y - size / 2), 0, self.precip.shape[1] - size))
        return self.precip[i, y0:y0 + size, x0:x0 + size].float().cpu().numpy()

    def save(self, path):
        np.savez_compressed(path, precip=self.precip.cpu().numpy(), time=self.time, lat=self.lat, lon=self.lon)

    @staticmethod
    def load(path):
        z = np.load(path)
        return FieldBank(torch.from_numpy(z["precip"]), z["time"], z["lat"], z["lon"])

    @staticmethod
    def from_array(precip, time=None, lat=None, lon=None):
        """Your own fields: precip (N, H, W) rain rate in mm/h with row 0 = north; time defaults to 6-hourly."""
        p = np.nan_to_num(np.asarray(precip, dtype=np.float32)).clip(0, None)
        n, h, w = p.shape
        if time is None:
            time = np.datetime64("2000-01-01T00", "s") + np.arange(n) * np.timedelta64(6, "h")
        time = np.asarray(time).astype("datetime64[s]")
        order = np.argsort(time)
        lat = np.arange(h, 0, -1, dtype=float) if lat is None else np.asarray(lat, dtype=float)
        lon = np.arange(w, dtype=float) if lon is None else np.asarray(lon, dtype=float)
        return FieldBank(torch.from_numpy(np.ascontiguousarray(p[order])).half(), time[order], lat, lon)


def download_era5(path, years=range(2016, 2024), months=range(5, 11), hours=(0, 6, 12, 18),
                  lat=(0.0, 60.0), lon=(90.0, 180.0), store=ARCO_ERA5, workers=32) -> FieldBank:
    """Hourly ERA5 total precipitation (mm/h) over a lat/lon box from the public ARCO-ERA5 store.

    Every chosen hour is one full-globe chunk on the server, so the time it takes depends on the
    number of hours, not on the box size. The result is cached at `path` (.npz).
    """
    if os.path.exists(path):
        return FieldBank.load(path)
    import pandas as pd
    import xarray as xr

    ds = xr.open_zarr(store, chunks=None, storage_options=dict(token="anon"))
    tp = ds["total_precipitation"]
    descending = tp["latitude"].values[0] > tp["latitude"].values[-1]
    tp = tp.sel(latitude=slice(max(lat), min(lat)) if descending else slice(min(lat), max(lat)),
                longitude=slice(min(lon), max(lon)))
    times = pd.DatetimeIndex(ds["time"].values)
    times = times[times.year.isin(list(years)) & times.month.isin(list(months)) & times.hour.isin(list(hours))]
    lat_v, lon_v = tp["latitude"].values, tp["longitude"].values
    out = np.zeros((len(times), len(lat_v), len(lon_v)), np.float16)

    def fetch(k):
        a = np.nan_to_num(tp.sel(time=times[k]).values * 1000.0).clip(0, None)  # m per hour -> mm/h
        out[k] = a if descending else a[::-1]

    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(fetch, range(len(times))), total=len(times), desc="ERA5 hours"))
    bank = FieldBank(torch.from_numpy(out), times.values.astype("datetime64[s]"),
                     lat_v if descending else lat_v[::-1], lon_v)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    bank.save(path)
    return bank


class FieldPairSynth:
    """Self-supervised forecast/observation pairs with exact labels, rendered on the GPU.

    A ("observation") is a canvas-sized crop of a field, turned/mirrored by multiples of 90 deg for
    variety. B ("forecast") shows the same field ('same' pairs) or an unrelated date and place
    ('other' pairs) through a known transform -- shift, zoom and rotation about the canvas centre --
    times an intensity factor, plus forecast-like errors of random strength: smooth warping, part of
    the pattern replaced, a changed intensity distribution, smoothing, spurious and missing cells,
    noise. Rain that moves into view comes from the surrounding real field.

    Labels: params (tx, ty, log_s, theta) and log_a (meaningful for 'same' pairs only), and
    pattern = correlation of smoothed log rain between B and A moved by the exact transform, i.e.
    how alike the two fields would be with B's shift / size / rotation / amount errors removed.
    """

    def __init__(self, bank: FieldBank, canvas=128, max_shift=0.25, scale_range=(0.7, 1.4), max_rot_deg=40.0,
                 amp_range=(0.5, 2.0), keep_probs=(0.2, 0.3, 0.4, 0.3), p_other=0.25, distortion=1.0,
                 min_wet=0.05, wet_mmh=0.2, other_min_days=5):
        self.bank, self.canvas = bank, canvas
        self.max_shift, self.scale_range, self.max_rot_deg = max_shift, scale_range, max_rot_deg
        self.amp_range, self.keep_probs = amp_range, keep_probs  # keep_probs: chance that shift / size /
        self.p_other, self.distortion = p_other, distortion      # rotation / amount stay exactly unchanged
        self.min_wet, self.wet_mmh, self.other_min_days = min_wet, wet_mmh, other_min_days
        self.device = bank.precip.device
        self._tsec = bank.time.astype("datetime64[s]").astype(np.int64)
        self._ident = F.affine_grid(torch.eye(2, 3, device=self.device)[None], [1, 1, canvas, canvas],
                                    align_corners=False)
        c = torch.arange(canvas, device=self.device, dtype=torch.float32) + 0.5
        self._yy, self._xx = torch.meshgrid(c, c, indexing="ij")

    def _rand(self, *shape, g=None):
        return torch.rand(shape, generator=g, device=self.device)

    def _randn(self, *shape, g=None):
        return torch.randn(shape, generator=g, device=self.device)

    def _centers(self, n, g):
        H, W = self.bank.precip.shape[-2:]
        r = self._rand(n, 2, g=g) - 0.5
        return torch.stack([W / 2 + r[:, 0] * max(W - self.canvas, 0), H / 2 + r[:, 1] * max(H - self.canvas, 0)], 1)

    def _base_matrix(self, center, d4):
        """Canvas -> source-field coordinates for crops centred at `center` (n, 2) px (x, y), turned by
        d4 (n,) in 0..7: quarter turns, plus a mirror for d4 >= 4."""
        n, L = center.shape[0], self.canvas
        H, W = self.bank.precip.shape[-2:]
        k = (d4 % 4).float() * (math.pi / 2)
        c, s = k.cos().round(), k.sin().round()
        f = 1.0 - 2.0 * (d4 >= 4).float()
        G = torch.zeros(n, 3, 3, device=self.device)
        G[:, 0, 0], G[:, 0, 1] = c * f * L / 2, -s * L / 2
        G[:, 1, 0], G[:, 1, 1] = s * f * L / 2, c * L / 2
        G[:, :2, 2] = center
        G[:, 2, 2] = 1
        to_norm = torch.tensor([[2 / W, 0, -1], [0, 2 / H, -1], [0, 0, 1]], device=self.device)
        return to_norm @ G

    def _render(self, idx, C, disp=None):
        """Sample fields idx at C q for every canvas point q (+ optional displacement), -> (n, 1, L, L) mm/h."""
        n, L = idx.shape[0], self.canvas
        q = self._ident.expand(n, L, L, 2)
        if disp is not None:
            q = q + disp
        grid = torch.einsum("nij,nhwj->nhwi", C[:, :2, :2], q) + C[:, None, None, :2, 2]
        src = self.bank.precip[idx].float().unsqueeze(1)
        return F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def _params(self, n, g):
        r = lambda: self._rand(n, g=g)
        lo, hi = math.log(self.scale_range[0]), math.log(self.scale_range[1])
        alo, ahi = math.log(self.amp_range[0]), math.log(self.amp_range[1])
        ang, rad = 2 * math.pi * r(), self.max_shift * r()
        p = torch.stack([rad * ang.cos(), rad * ang.sin(), lo + (hi - lo) * r(),
                         math.radians(self.max_rot_deg) * (2 * r() - 1)], 1)
        log_a = alo + (ahi - alo) * r()
        keep = self._rand(4, n, g=g) < torch.tensor(self.keep_probs, device=self.device)[:, None]
        p = p * torch.stack([~keep[0], ~keep[0], ~keep[1], ~keep[2]], 1).float()
        return p, log_a * (~keep[3]).float()

    def _blob(self, cx, cy, rad):
        return torch.exp(-((self._xx - cx) ** 2 + (self._yy - cy) ** 2) / (2 * rad ** 2))

    def _distort(self, r, u, mix, g):
        """Forecast-like errors of strength u (n,) in [0, 1]; mix = an unrelated field to paste in."""
        n, L = r.shape[0], self.canvas
        uu = u.view(n, 1, 1, 1)
        rnd = lambda: self._rand(n, 1, 1, 1, g=g)
        up = lambda x: F.interpolate(x, size=(L, L), mode="bilinear", align_corners=False)
        # part of the pattern replaced by an unrelated one
        cover = 0.6 * uu * rnd()
        w = ((up(self._rand(n, 1, 3, 3, g=g)) - (1 - cover)) / 0.15).clamp(0, 1)
        r = (1 - w) * r + w * mix
        # changed intensity distribution, same mean
        gamma = torch.exp(0.25 * uu * self._randn(n, 1, 1, 1, g=g))
        rg = r.clamp_min(0) ** gamma
        r = rg * r.mean((2, 3), keepdim=True) / rg.mean((2, 3), keepdim=True).clamp_min(1e-6)
        # smoothing
        r = gaussian_blur(r, 1.5 * u * self._rand(n, g=g))
        # spurious cells anywhere, missing cells where it rains
        peak = r.amax((2, 3), keepdim=True)
        for _ in range(3):
            on = (rnd() < 0.5 * uu).float()
            r = r + on * (0.2 + 0.6 * rnd()) * peak * self._blob(rnd() * L, rnd() * L, 1.5 + 2.5 * rnd())
        for _ in range(2):
            on = (rnd() < 0.5 * uu).float()
            k = torch.multinomial(r.flatten(1) + 1e-6, 1, generator=g).view(n, 1, 1, 1)
            r = r * (1 - on * self._blob((k % L).float() + 0.5, (k // L).float() + 0.5, 3 + 5 * rnd()))
        # pixel noise
        r = r * torch.exp(0.3 * uu * self._randn(n, 1, L, L, g=g))
        return r.clamp_min(0)

    def sample(self, n, rng=None, g=None):
        """A batch of n labelled pairs (dict of tensors on the bank's device). Pass a seeded numpy
        `rng` and a torch generator `g` on the bank's device to get the same batch every time."""
        rng = rng if rng is not None else np.random.default_rng()
        dev, N, L = self.device, len(self.bank), self.canvas

        # observations: oversample crops, prefer ones with enough rain
        m = 2 * n
        idx = torch.as_tensor(rng.integers(0, N, m), device=dev)
        center = self._centers(m, g)
        d4 = torch.randint(0, 8, (m,), generator=g, device=dev)
        G = self._base_matrix(center, d4)
        a = self._render(idx, G)
        dry = ((a > self.wet_mmh).float().mean((1, 2, 3)) < self.min_wet).float()
        keep = torch.argsort(dry, stable=True)[:n]
        idx, d4, G, a = idx[keep], d4[keep], G[keep], a[keep]

        # forecast source: the same field, or an unrelated date and place
        j = idx.cpu().numpy().copy()
        other = np.zeros(n, bool)
        for k in np.flatnonzero(rng.random(n) < self.p_other):
            for _ in range(20):
                c = rng.integers(0, N)
                if abs(self._tsec[c] - self._tsec[j[k]]) >= self.other_min_days * 86400:
                    j[k], other[k] = c, True
                    break
        other = torch.as_tensor(other, device=dev)
        j = torch.as_tensor(j, device=dev)
        G_other = self._base_matrix(self._centers(n, g), torch.randint(0, 8, (n,), generator=g, device=dev))
        G_b = torch.where(other[:, None, None], G_other, G)

        # exact transform + intensity, then forecast-like errors
        params, log_a = self._params(n, g)
        Minv = torch.linalg.inv(sim_matrix(params))
        amp = log_a.exp().view(n, 1, 1, 1)
        u = self.distortion * self._rand(n, g=g) * (self._rand(n, g=g) > 0.2).float()
        disp = F.interpolate(self._randn(n, 2, 5, 5, g=g), size=(L, L), mode="bicubic", align_corners=False)
        disp = disp.permute(0, 2, 3, 1) * (u * 8 / L).view(n, 1, 1, 1)  # up to ~4 px
        ideal = amp * self._render(idx, G @ Minv)  # A moved exactly by the transform
        mix_idx = torch.as_tensor(rng.integers(0, N, n), device=dev)
        mix = amp * self._render(mix_idx, self._base_matrix(self._centers(n, g), d4) @ Minv)
        b = self._distort(amp * self._render(j, G_b @ Minv, disp), u, mix, g)
        return dict(a=a, b=b, ideal=ideal, params=params, log_a=log_a, pattern=smooth_corr(ideal, b),
                    same=~other, u=u)

    def make_pair(self, i, center=None, shift_px=(0.0, 0.0), scale=1.0, rotation_ccw_deg=0.0, intensity=1.0, j=None):
        """A deterministic, undistorted (obs, forecast) pair for demos, as (L, L) tensors in mm/h.

        obs = field i cropped around `center` ((x, y) px of the field, default: middle); forecast =
        field j (default i) shifted by shift_px = (east, north) grid points, zoomed by `scale`,
        rotated counter-clockwise, and multiplied by `intensity`.
        """
        dev, L = self.device, self.canvas
        H, W = self.bank.precip.shape[-2:]
        cx, cy = center if center is not None else (W / 2, H / 2)
        G = self._base_matrix(torch.tensor([[float(cx), float(cy)]], device=dev), torch.zeros(1, dtype=torch.long, device=dev))
        p = torch.tensor([[shift_px[0] / L, -shift_px[1] / L, math.log(scale), -math.radians(rotation_ccw_deg)]], device=dev)
        Minv = torch.linalg.inv(sim_matrix(p))
        obs = self._render(torch.tensor([i], device=dev), G)
        fcst = intensity * self._render(torch.tensor([i if j is None else j], device=dev), G @ Minv)
        return obs[0, 0], fcst[0, 0]
