"""Oxford-IIIT Pet -> in-memory tile bank, and a GPU synthesizer of labelled image pairs.

Pairs are rendered on the fly with ``grid_sample``, so the centre and size of the animal
in every image are known exactly and every training batch is new.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

# The 12 cat breeds of Oxford-IIIT Pet, as torchvision names them; the other 25 breeds are dogs.
CAT_BREEDS = frozenset({
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
    "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx",
})

# How the two images of a pair are related, from most to least similar.
RELATIONS = ("same_instance", "same_breed", "same_species", "diff_species")


@dataclass
class TileBank:
    """One dataset split, each photo letterboxed into a square RGBA tile (alpha = animal mask).

    Coordinates are in the tile's normalised frame: [-1, 1] on both axes, as in grid_sample.
    """

    tiles: torch.Tensor    # uint8 (N, 4, T, T)
    bbox: torch.Tensor     # float (N, 4): animal bounding box (cx, cy, w, h)
    rect: torch.Tensor     # float (N, 4): photo area (x0, y0, x1, y1), i.e. without letterbox padding
    breed: torch.Tensor    # long (N,)
    species: torch.Tensor  # long (N,): 0 = cat, 1 = dog
    classes: list

    def __len__(self):
        return self.tiles.shape[0]

    def to(self, device):
        t = lambda x: x.to(device)
        return TileBank(t(self.tiles), t(self.bbox), t(self.rect), t(self.breed), t(self.species), self.classes)

    def save(self, path):
        torch.save({k: getattr(self, k) for k in ("tiles", "bbox", "rect", "breed", "species", "classes")}, path)

    @staticmethod
    def load(path):
        return TileBank(**torch.load(path, map_location="cpu"))


def make_tile(image: Image.Image, trimap: Image.Image, size: int):
    """Letterbox one photo into a size x size RGBA tile; returns (tile, bbox, rect) or None.

    Oxford-IIIT trimaps: 1 = animal, 2 = background, 3 = boundary (kept as half alpha).
    """
    image = image.convert("RGB")
    w, h = image.size
    r = size / max(w, h)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    rgb = np.asarray(image.resize((nw, nh), Image.Resampling.BILINEAR))
    tri = np.asarray(trimap.resize((nw, nh), Image.Resampling.NEAREST))
    alpha = np.select([tri == 1, tri == 3], [255, 128], 0).astype(np.uint8)
    ys, xs = np.nonzero(alpha)
    if len(xs) < 16:
        return None
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    tile = np.zeros((4, size, size), np.uint8)
    tile[:3, y0:y0 + nh, x0:x0 + nw] = rgb.transpose(2, 0, 1)
    tile[3, y0:y0 + nh, x0:x0 + nw] = alpha
    n = lambda p: 2.0 * p / size - 1.0  # pixel edge -> normalised coordinate
    bx0, bx1 = n(x0 + xs.min()), n(x0 + xs.max() + 1)
    by0, by1 = n(y0 + ys.min()), n(y0 + ys.max() + 1)
    bbox = ((bx0 + bx1) / 2, (by0 + by1) / 2, bx1 - bx0, by1 - by0)
    rect = (n(x0 + 1), n(y0 + 1), n(x0 + nw - 1), n(y0 + nh - 1))  # 1 px inset hides resampled black edges
    return tile, bbox, rect


def bank_from_samples(samples, classes) -> TileBank:
    """samples: list of (tile, bbox, rect, breed_label) with tile/bbox/rect from make_tile."""
    tiles, bbox, rect, breed = zip(*samples)
    is_cat = torch.tensor([c in CAT_BREEDS for c in classes])
    breed = torch.tensor(breed, dtype=torch.long)
    return TileBank(
        tiles=torch.from_numpy(np.stack(tiles)),
        bbox=torch.tensor(bbox, dtype=torch.float32),
        rect=torch.tensor(rect, dtype=torch.float32),
        breed=breed,
        species=(~is_cat[breed]).long(),
        classes=list(classes),
    )


def build_bank(root, split="trainval", tile=160, cache_dir=None, download=True, workers=8) -> TileBank:
    """Oxford-IIIT Pet split ('trainval' or 'test') as a TileBank; downloads ~800 MB on first use."""
    cache = os.path.join(cache_dir, f"pets_{split}_{tile}.pt") if cache_dir else None
    if cache and os.path.exists(cache):
        return TileBank.load(cache)

    from torchvision.datasets import OxfordIIITPet

    ds = OxfordIIITPet(root, split=split, target_types=["category", "segmentation"], download=download)
    if sum(c in CAT_BREEDS for c in ds.classes) != len(CAT_BREEDS):
        raise RuntimeError(f"unexpected Oxford-IIIT Pet class names: {ds.classes}")

    def load(i):
        image, (label, trimap) = ds[i]
        out = make_tile(image, trimap, tile)
        return None if out is None else (*out, label)

    with ThreadPoolExecutor(workers) as ex:
        samples = [s for s in tqdm(ex.map(load, range(len(ds))), total=len(ds), desc=f"tiles[{split}]")
                   if s is not None]
    bank = bank_from_samples(samples, ds.classes)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        bank.save(cache)
    return bank


class PairSynth:
    """Renders random image pairs with exact labels for pattern, location and scale.

    Each image shows one animal whose bounding-box centre is at ``center`` ((x, y) as
    fractions of the canvas, y pointing down) and whose longest box side is ``size`` x the
    canvas side. The animal is either cut out and pasted on a procedural background
    ("cutout") or shown inside its original photo ("photo").

    Pattern targets come from how the two animals are related (``RELATIONS``), e.g. by
    default 1.0 for the same animal, 0.75 same breed, 0.4 same species, 0.0 cat vs dog.
    """

    def __init__(self, bank: TileBank, canvas=160, size_range=(0.25, 0.7), p_photo=0.5,
                 relation_probs=(0.3, 0.25, 0.2, 0.25), pattern_levels=(1.0, 0.75, 0.4, 0.0),
                 geometry_probs=(0.2, 0.3, 0.5), jitter=(0.08, 0.15), color_jitter=0.15):
        self.bank, self.canvas, self.size_range, self.p_photo = bank, canvas, size_range, p_photo
        self.relation_probs, self.geometry_probs = relation_probs, geometry_probs
        self.jitter, self.color_jitter = jitter, color_jitter  # jitter = (centre std, log-size std)
        self.device = bank.tiles.device
        self.pattern_levels = torch.tensor(pattern_levels, device=self.device)
        self._breed = bank.breed.cpu().numpy()
        self._species = bank.species.cpu().numpy()
        self._by_breed = {b: np.flatnonzero(self._breed == b) for b in np.unique(self._breed)}
        self._breeds_of = {s: np.unique(self._breed[self._species == s]) for s in (0, 1)}

    def _rand(self, *shape, g=None):
        return torch.rand(shape, generator=g, device=self.device)

    def _randn(self, *shape, g=None):
        return torch.randn(shape, generator=g, device=self.device)

    def _partners(self, a, rng):
        """For each index in a, draw a partner that is the same animal / breed / species / other species."""
        b = a.copy()
        for i, r in enumerate(rng.choice(4, size=len(a), p=self.relation_probs)):
            breed, species = self._breed[a[i]], self._species[a[i]]
            if r == 0:
                continue
            if r == 1:
                breeds = [breed]
            elif r == 2:
                breeds = [x for x in self._breeds_of[species] if x != breed] or [breed]
            else:
                breeds = list(self._breeds_of[1 - species]) or [breed]
            b[i] = rng.choice(self._by_breed[rng.choice(breeds)])
        return b

    def relation(self, a, b):
        """Index into RELATIONS for each pair, from the actual indices drawn."""
        rel = np.full(len(a), 3, dtype=np.int64)
        rel[self._species[a] == self._species[b]] = 2
        rel[self._breed[a] == self._breed[b]] = 1
        rel[a == b] = 0
        return rel

    def _geometry(self, n, g):
        """Centres/sizes for A and B. B is either identical to A, a small jitter of A, or independent."""
        smin, smax = self.size_range
        lo, hi = math.log(smin), math.log(smax)
        size = lambda: torch.exp(lo + (hi - lo) * self._rand(n, g=g))
        place = lambda s: s[:, None] / 2 + self._rand(n, 2, g=g) * (1 - s[:, None])  # box stays inside
        size_a = size()
        center_a = place(size_a)
        size_b = size()
        center_b = place(size_b)
        size_j = (size_a * torch.exp(self.jitter[1] * self._randn(n, g=g))).clamp(smin, smax)
        half = size_j[:, None] / 2
        center_j = torch.minimum(torch.maximum(center_a + self.jitter[0] * self._randn(n, 2, g=g), half), 1 - half)
        probs = torch.tensor(self.geometry_probs, device=self.device)
        mode = torch.multinomial(probs, n, replacement=True, generator=g)
        same, jit = mode == 0, mode == 1
        size_b = torch.where(same, size_a, torch.where(jit, size_j, size_b))
        center_b = torch.where(same[:, None], center_a, torch.where(jit[:, None], center_j, center_b))
        return center_a, size_a, center_b, size_b

    def background(self, n, g=None):
        """Random smooth colour fields with blobs and grain."""
        H = self.canvas
        r = lambda *s: self._rand(*s, g=g)
        up = lambda x: F.interpolate(x, size=(H, H), mode="bilinear", align_corners=False)
        bg = 0.2 + 0.6 * r(n, 3, 1, 1)
        bg = bg + 0.8 * r(n, 1, 1, 1) * (up(r(n, 3, 3, 3)) - 0.5)
        bg = bg + 0.5 * r(n, 1, 1, 1) * (up(r(n, 3, 12, 12)) - 0.5)
        bg = bg + 0.2 * r(n, 1, 1, 1) * (r(n, 3, H, H) - 0.5)
        gray = r(n, 1, 1, 1) < 0.2
        bg = torch.where(gray, bg.mean(1, keepdim=True).expand_as(bg), bg)
        return bg.clamp(0, 1)

    def render(self, idx, center, size, photo=False, flip=False, g=None):
        """Draw tiles ``idx`` with box centre ``center`` (n, 2) and size ``size`` (n,).

        Returns images (n, 3, H, W) in [0, 1] and the animals' masks (n, 1, H, W).
        """
        dev, H = self.device, self.canvas
        idx = torch.as_tensor(idx, device=dev).long()
        n = idx.numel()
        center = torch.as_tensor(center, dtype=torch.float32, device=dev).expand(n, 2)
        size = torch.as_tensor(size, dtype=torch.float32, device=dev).expand(n)
        photo = torch.as_tensor(photo, device=dev).bool().expand(n)
        flip = torch.as_tensor(flip, device=dev).bool().expand(n)
        tiles = self.bank.tiles[idx].float() / 255
        bbox, rect = self.bank.bbox[idx], self.bank.rect[idx]

        # Canvas -> tile affine map, both in [-1, 1] coords: x_tile = bbox_centre + k * (x_canvas - c).
        k = torch.maximum(bbox[:, 2], bbox[:, 3]) / (2 * size)
        kx = torch.where(flip, -k, k)
        c = 2 * center - 1
        theta = torch.zeros(n, 2, 3, device=dev)
        theta[:, 0, 0], theta[:, 0, 2] = kx, bbox[:, 0] - kx * c[:, 0]
        theta[:, 1, 1], theta[:, 1, 2] = k, bbox[:, 1] - k * c[:, 1]
        grid = F.affine_grid(theta, [n, 4, H, H], align_corners=False)
        warped = F.grid_sample(tiles, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        rgb, mask = warped[:, :3], warped[:, 3:]

        gx, gy = grid[..., 0], grid[..., 1]
        r = rect[:, :, None, None]
        in_photo = ((gx >= r[:, 0]) & (gx <= r[:, 2]) & (gy >= r[:, 1]) & (gy <= r[:, 3])).unsqueeze(1).float()
        cover = torch.where(photo[:, None, None, None], in_photo, mask)
        img = cover * rgb + (1 - cover) * self.background(n, g)
        return img, mask

    def _photometric(self, img, g):
        if self.color_jitter <= 0:
            return img
        n, j = img.shape[0], self.color_jitter
        u = lambda: 1 + j * (2 * self._rand(n, 1, 1, 1, g=g) - 1)
        mean = img.mean(dim=(1, 2, 3), keepdim=True)
        img = (img - mean) * u() + mean * u()  # contrast, brightness
        img = img + 0.02 * self._randn(*img.shape, g=g)
        return img.clamp(0, 1)

    def sample(self, n, rng=None, g=None):
        """A batch of n labelled pairs (dict of tensors on the bank's device).

        Pass a seeded numpy ``rng`` and a torch generator ``g`` on the bank's device to get
        the same batch every time.
        """
        rng = rng if rng is not None else np.random.default_rng()
        a = rng.integers(0, len(self.bank), n)
        b = self._partners(a, rng)
        rel = torch.as_tensor(self.relation(a, b), device=self.device)
        center_a, size_a, center_b, size_b = self._geometry(n, g)
        flip = self._rand(n, g=g) < 0.5  # mirror both images of a pair alike
        photo_a, photo_b = self._rand(2, n, g=g) < self.p_photo
        img_a, mask_a = self.render(a, center_a, size_a, photo_a, flip, g)
        img_b, mask_b = self.render(b, center_b, size_b, photo_b, flip, g)
        a, b = torch.as_tensor(a, device=self.device), torch.as_tensor(b, device=self.device)
        return dict(
            img_a=self._photometric(img_a, g), img_b=self._photometric(img_b, g),
            mask_a=mask_a, mask_b=mask_b,
            center_a=center_a, center_b=center_b, size_a=size_a, size_b=size_b,
            pattern=self.pattern_levels[rel], relation=rel,
            breed_a=self.bank.breed[a], breed_b=self.bank.breed[b], idx_a=a, idx_b=b,
        )
