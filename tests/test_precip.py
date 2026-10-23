"""Fast CPU checks for precipsim on fake rain fields (no download):   python -m pytest -q tests"""
import math
import os
import sys

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from precipsim.classic import classic_register  # noqa: E402
from precipsim.data import FieldBank, FieldPairSynth  # noqa: E402
from precipsim.evaluate import evaluate, summary_table  # noqa: E402
from precipsim.geometry import resample, sim_matrix, sim_params  # noqa: E402
from precipsim.infer import FieldScorer  # noqa: E402
from precipsim.scores import SCORES  # noqa: E402
from precipsim.train import Config, train  # noqa: E402
from precipsim.viz import show_pairs, show_result  # noqa: E402


def blob_field(rng, h, w, k=25):
    """Smooth fake rain field (mm/h): a sum of elongated Gaussian cells."""
    yy, xx = np.mgrid[0:h, 0:w]
    f = np.zeros((h, w))
    for _ in range(k):
        cy, cx, sy, sx = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(2, 10), rng.uniform(2, 10)
        f += rng.uniform(0.5, 8) * np.exp(-((yy - cy) / sy) ** 2 / 2 - ((xx - cx) / sx) ** 2 / 2)
    return f


@pytest.fixture(scope="module")
def bank():
    rng = np.random.default_rng(0)
    return FieldBank.from_array(np.stack([blob_field(rng, 80, 100) for _ in range(24)]))


def test_matrix_roundtrip():
    p = torch.tensor([[0.1, -0.05, 0.2, 0.3], [0.0, 0.0, 0.0, 0.0], [-0.2, 0.1, -0.3, -0.6]])
    assert torch.allclose(sim_params(sim_matrix(p)), p, atol=1e-5)
    M = sim_matrix(p)
    assert torch.allclose(sim_params(M @ torch.linalg.inv(M)), torch.zeros(3, 4), atol=1e-5)


def test_forecast_is_transformed_observation(bank):
    synth = FieldPairSynth(bank, canvas=64, p_other=0.0, distortion=0.0, keep_probs=(0, 0, 0, 0))
    batch = synth.sample(8, np.random.default_rng(0), torch.Generator().manual_seed(0))
    assert torch.allclose(batch["pattern"], torch.ones(8), atol=1e-4)
    M = sim_matrix(batch["params"])
    back = resample(batch["b"], M)  # B(M q) should equal amount * A(q)
    inside = resample(torch.ones_like(batch["b"]), M) > 0.999
    inside[..., :3, :] = inside[..., -3:, :] = False
    inside[..., :, :3] = inside[..., :, -3:] = False
    expect = batch["log_a"].exp().view(-1, 1, 1, 1) * batch["a"]
    rel = ((back - expect).abs() * inside).sum() / (expect.abs() * inside).sum()
    assert rel < 0.08


def test_pairs(bank):
    synth = FieldPairSynth(bank, canvas=64, other_min_days=2)
    b1 = synth.sample(6, np.random.default_rng(5), torch.Generator().manual_seed(5))
    b2 = synth.sample(6, np.random.default_rng(5), torch.Generator().manual_seed(5))
    assert torch.equal(b1["b"], b2["b"])
    assert b1["a"].shape == b1["b"].shape == (6, 1, 64, 64)
    assert float(b1["b"].min()) >= 0 and torch.all((b1["pattern"] >= 0) & (b1["pattern"] <= 1))
    show_pairs(b1, n=3)


def test_classic_baseline_recovers_transform(bank):
    synth = FieldPairSynth(bank, canvas=64, p_other=0.0, distortion=0.0, max_shift=0.15,
                           scale_range=(0.8, 1.25), max_rot_deg=20)
    batch = synth.sample(8, np.random.default_rng(1), torch.Generator().manual_seed(1))
    e = classic_register(batch["a"], batch["b"])["params"] - batch["params"]
    assert float((e[:, :2].norm(dim=1) * 64).median()) < 3
    assert float(e[:, 2].abs().median()) < 0.1
    assert float(torch.rad2deg(e[:, 3].abs()).median()) < 7


def test_train_evaluate_and_score(bank, tmp_path):
    cfg = Config(out_dir=str(tmp_path), canvas=64, steps=4, batch_size=4, eval_every=2, eval_pairs=8,
                 log_every=2, pretrained=False, amp=False, warmup=2)
    model, history = train(cfg, bank, bank, device="cpu")
    assert len(history) == 2 and all(math.isfinite(h["loss"]) for h in history)
    m = evaluate(model, FieldPairSynth(bank, canvas=64), n_pairs=8, baseline=True)
    assert {"shift_err_px", "base_shift_err_px", "moments_shift_err_px", "rawcorr_pattern_mae"} <= set(m)
    summary_table(m)

    scorer = FieldScorer.from_checkpoint(os.path.join(tmp_path, "precipsim.pt"), device="cpu")
    obs = blob_field(np.random.default_rng(3), 50, 70)
    r = scorer.score(obs, np.roll(obs, 4, axis=1), grid_km=10, return_maps=True)
    for k in SCORES:
        assert 0.0 <= r[k] <= 1.0
    assert r["maps"]["obs"].shape == (64, 64) and "shift_km" in r
    rb = scorer.score_batch(np.stack([obs, obs]), np.stack([obs, obs * 2]))
    assert rb["pattern"].shape == (2,)
    with pytest.raises(ValueError):
        scorer.score(obs, obs[:40])
    show_result(r, title="test")
