"""Fast CPU checks on a fake dataset (no download):   python -m pytest -q tests"""
import math
import os
import sys

import matplotlib
import numpy as np
import pytest
import torch
from PIL import Image

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patsim.data import PairSynth, bank_from_samples, make_tile  # noqa: E402
from patsim.infer import PairScorer  # noqa: E402
from patsim.train import Config, train  # noqa: E402
from patsim.viz import show_batch, show_result  # noqa: E402

CLASSES = ["Bengal", "Persian", "Beagle", "Pug"]  # two cat breeds, two dog breeds


def fake_photo(rng, label):
    """A random-size 'photo' whose 'animal' is a coloured rectangle, plus its trimap."""
    w, h = rng.integers(60, 120, 2)
    img = np.zeros((h, w, 3), np.uint8) + rng.integers(0, 255, 3).astype(np.uint8)
    tri = np.full((h, w), 2, np.uint8)
    bw, bh = rng.integers(w // 4, w // 2), rng.integers(h // 4, h // 2)
    x0, y0 = rng.integers(0, w - bw), rng.integers(0, h - bh)
    img[y0:y0 + bh, x0:x0 + bw] = (60 * (label + 1), 255 - 50 * label, 30)
    tri[y0:y0 + bh, x0:x0 + bw] = 3
    tri[y0 + 1:y0 + bh - 1, x0 + 1:x0 + bw - 1] = 1
    return Image.fromarray(img), Image.fromarray(tri)


@pytest.fixture(scope="module")
def bank():
    rng = np.random.default_rng(0)
    samples = [(*make_tile(*fake_photo(rng, i % 4), size=64), i % 4) for i in range(32)]
    return bank_from_samples(samples, CLASSES)


def test_bank(bank):
    assert bank.tiles.shape == (32, 4, 64, 64) and bank.tiles.dtype == torch.uint8
    assert bank.species.tolist()[:4] == [0, 0, 1, 1]


def test_render_places_object_exactly(bank):
    S = 96
    synth = PairSynth(bank, canvas=S)
    center = torch.tensor([[0.3, 0.3], [0.7, 0.6], [0.5, 0.5], [0.4, 0.65]] * 2)
    size = torch.tensor([0.3, 0.45, 0.6, 0.35] * 2)
    flip = torch.tensor([False] * 4 + [True] * 4)
    for photo in (False, True):
        _, mask = synth.render(torch.arange(8), center, size, photo=photo, flip=flip)
        for m, c, s in zip(mask[:, 0], center, size):
            ys, xs = torch.nonzero(m > 0.25, as_tuple=True)
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            # tolerances cover pixel quantisation + bilinear blur of small upscaled tiles;
            # a wrong affine map is off by tens of pixels
            assert abs((x0 + x1) / 2 / S - c[0]) < 2.5 / S
            assert abs((y0 + y1) / 2 / S - c[1]) < 2.5 / S
            assert abs(max(x1 - x0, y1 - y0) / S - s) < 6 / S


def test_sample_labels(bank):
    synth = PairSynth(bank, canvas=64)
    batch = synth.sample(16, np.random.default_rng(1))
    assert batch["img_a"].shape == (16, 3, 64, 64) and batch["mask_b"].shape == (16, 1, 64, 64)
    assert 0 <= float(batch["img_a"].min()) and float(batch["img_b"].max()) <= 1
    same = batch["idx_a"] == batch["idx_b"]
    assert torch.all(batch["pattern"][same] == 1.0)
    assert torch.all(batch["pattern"][~same] < 1.0)
    for s in "ab":
        half = batch[f"size_{s}"][:, None] / 2
        c = batch[f"center_{s}"]
        assert torch.all(c >= half - 1e-6) and torch.all(c <= 1 - half + 1e-6)
    # same seeds -> same batch
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    b1 = synth.sample(4, np.random.default_rng(5), g1)
    b2 = synth.sample(4, np.random.default_rng(5), g2)
    assert torch.equal(b1["img_a"], b2["img_a"])
    show_batch(batch, n=3)


def test_train_and_score(bank, tmp_path):
    cfg = Config(out_dir=str(tmp_path), canvas=64, steps=4, batch_size=4, eval_every=2, eval_pairs=8,
                 log_every=2, pretrained=False, amp=False, warmup=2)
    _, history = train(cfg, bank, bank, device="cpu")
    assert len(history) == 2 and all(math.isfinite(h["loss"]) for h in history)

    scorer = PairScorer.from_checkpoint(os.path.join(tmp_path, "patsim.pt"), device="cpu")
    img = np.random.default_rng(0).integers(0, 255, (50, 80, 3), dtype=np.uint8)
    same = scorer.score(img, img, return_maps=True)
    assert same["location"] == pytest.approx(1.0) and same["scale"] == pytest.approx(1.0)
    show_result(same, title="same")
    field = np.random.default_rng(1).random((70, 40))  # single-channel float "field"
    other = scorer.score(img, field)
    for k in ("pattern", "location", "scale", "overall"):
        assert 0.0 <= other[k] <= 1.0
    batch = scorer.score_canvases(torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64))
    assert batch["pattern"].shape == (2,)
