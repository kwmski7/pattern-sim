# pattern-sim

Learned similarity between two 2D fields that keeps **what** the pattern is apart from **where**,
**how big**, **how rotated** and **how intense** it is.

* **`precipsim/`**: precipitation forecast vs. observation, with no objects or bounding boxes. This is the main package.
  Notebook: [`notebooks/colab_precip.ipynb`](notebooks/colab_precip.ipynb)
* **`patsim/`**: the first prototype, on cat/dog photos with per-image object boxes.
  Notebook: [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb)

## precipsim

| output | meaning | perfect forecast |
|---|---|---|
| `pattern` | correlation of smoothed log rain **after** shift/size/rotation/amount errors are removed | 1 |
| `location` | `1 / (1 + d/d0)`, `d` = shift as a fraction of the domain, `d0 = 0.1` | 1 |
| `scale` | smaller / larger size | 1 |
| `rotation` | `1 / (1 + angle/20°)` | 1 |
| `intensity` | smaller / larger rain amount | 1 |
| `overall` | geometric mean of the five | 1 |
| `shift_east/north` (grid pts, `shift_km`), `scale_ratio`, `rotation_deg` (ccw), `intensity_ratio` | the estimated errors themselves | 0, 1, 0, 1 |
| `unc_shift`, `unc_scale_pct`, `unc_rotation_deg` | the network's expected error of those estimates | |
| `corr_raw`, `centroid_*`, `amount_ratio` | classical references: gridpoint correlation and SAL-style moments | |

### How it works

```
 obs A ──┐                                        ┌─► shift, zoom, rotation (+ uncertainty)
         ├─ shared ResNet-18 ─► all-pairs feature ┤
 fcst B ─┘   (stride 8)         correlation       └─► move B back onto A ─► compare CNN ─► pattern, amount ratio
```

1. **Registration:** every location in A's feature map is correlated with every location in B's. A small
   CNN turns that correlation volume into the similarity transform from A to B (in the style of Rocco et al. 2017,
   *CNN architecture for geometric matching*). At inference, one extra pass refines it.
2. **Comparison:** B is warped back by that transform, and a second CNN compares it with A to give the
   pattern score and the rain-amount ratio.

There is no notion of an "object", so typhoons, fronts and scattered convection are handled the same way.

### Training data: self-supervised, exact labels

`FieldPairSynth` takes a real field (ERA5 by default, or your own observations) and does the following:

* **Observation A:** a 128 × 128 crop. **Forecast B:** the same field seen through a random known
  shift (≤ 25 % of the domain), zoom (0.7–1.4), rotation (±40°) and amount factor (0.5–2). Rain that
  moves into view comes from the real surrounding field.
* **Forecast-like errors of random strength:** smooth warping, part of the pattern replaced, a changed
  intensity distribution, smoothing, spurious and missing cells, and noise. 25 % of pairs use an unrelated
  date and place as the forecast.
* **Labels:** the transform and amount factor are exact. The pattern target is the correlation
  between B and A moved by the exact transform.

The network is trained with smooth-L1 on the transform and amount, a Laplace NLL for the uncertainty, and BCE on
the pattern. Evaluation uses a held-out year and compares the network with a **brute-force classical
registration** (zoom/rotation grid + phase correlation) and **SAL-style moments** (`precipsim/classic.py`).

### Data

ERA5 hourly total precipitation from Google's public ARCO-ERA5 store
(`gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`, no login), 0–60°N,
90–180°E, May–October, 00/06/12/18 UTC. Train on 2016–2022, test on 2023.

### Use

```python
from precipsim import FieldScorer
scorer = FieldScorer.from_checkpoint("runs/precip/precipsim.pt")
r = scorer.score(obs, fcst, accum_hours=1, grid_km=12)   # 2D arrays, same grid, row 0 = north
r["pattern"], r["location"], r["shift_km"], r["rotation_deg"], r["intensity_ratio"]
scorer.score_batch(obs_stack, fcst_stack)                  # many cases at once -> arrays
```

To adapt to your own grid (radar, GPM, KIM), use `FieldBank.from_array(obs_mmh, time=...)` and then `train(...)`.
No labels are needed.

### Limitations

* Rotation and zoom are ill-defined for round or scattered rain. The uncertainty outputs show this.
* A single global transform is estimated. If two systems move differently, the result is a compromise, and the pattern
  score drops.
* ERA5 is smoother than radar. For km-scale convection, retrain on km-scale observations.

## Running

**Colab:** open a notebook, choose a T4 GPU runtime, set `GITHUB_USER`, then run all cells. You can also upload
`pattern-sim.zip` to `/content` instead of cloning.

**Locally:**

```bash
pip install -r requirements.txt
python -m pytest -q tests
python -m precipsim.train --data_path era5_ea.npz --steps 8000
```

## Files

```
precipsim/geometry.py   similarity transforms + warping (one convention everywhere)
precipsim/data.py       ERA5 download, FieldBank, FieldPairSynth (synthetic forecasts, exact labels)
precipsim/model.py      RegNet: correlation-volume registration + comparison CNN
precipsim/classic.py    baselines: exhaustive search + phase correlation, SAL-style moments
precipsim/scores.py     score definitions (d0, rot0_deg)
precipsim/train.py      training loop + CLI
precipsim/evaluate.py   fixed-seed test metrics, network vs. baselines
precipsim/infer.py      FieldScorer for arbitrary obs/forecast arrays
precipsim/viz.py        plots
patsim/                 cat/dog prototype (object boxes)
tests/                  CPU self-tests on fake data
notebooks/              Colab notebooks
```
