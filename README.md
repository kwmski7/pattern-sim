# pattern-sim

A deep-learning similarity measure for two 2D images that separates **what** is in them from
**where** and **how big** it is.

| output | meaning | identical images |
|---|---|---|
| `pattern` | same thing? (default targets: same animal 1, same breed 0.75, same species 0.4, cat vs dog 0) | 1 |
| `location` | `1 / (1 + d/d0)`, `d` = distance between object centres as a fraction of image size, `d0 = 0.1` | 1 |
| `scale` | `exp(-|log(size_B/size_A)|)` = smaller / larger | 1 |
| `overall` | geometric mean of the three | 1 |
| `dx, dy, scale_ratio` | raw displacement and size ratio of B relative to A | 0, 0, 1 |

Examples: the same cat moved from top-left to bottom-right keeps `pattern ≈ 1` while `location`
falls to ~0.15; a dog at the cat's place keeps `location ≈ 1` with low `pattern`; the same cat at
twice the size gives `scale ≈ 0.5`.

[Open the notebook in Colab](https://colab.research.google.com/github/YOUR_GITHUB_ID/pattern-sim/blob/main/notebooks/colab_train.ipynb)
(replace `YOUR_GITHUB_ID` once the repo is on GitHub).

## How it works

```
 image A ─┐                       ┌─ mask head ─► soft object mask ─► mask-pooled features ─► embedding e_A
          ├─ shared ResNet-18 ─► F├─ heat head ─► soft-argmax ───────► centre c_A (x, y)
 image B ─┘  (layer3 + layer4,    └─ size head ─────────────────────► log size s_A
              stride 16)
 pattern  = sigmoid(a · cos(e_A, e_B) + b)        (embedding pooled over the object only → blind to position/size)
 location = 1 / (1 + |c_B − c_A| / d0)            (soft-argmax → moves exactly with the object)
 scale    = exp(−|s_B − s_A|)
```

The three scores come from different parts of the network, so the pattern score is not affected
by position or size, and location and size are measured for each image separately (so a dog can still
be "in the same place" as a cat).

### Training data (Oxford-IIIT Pet, downloaded automatically)

[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) has 7,349 photos of 37 cat and dog breeds, each with
a pixel mask of the animal. Each photo becomes a square RGBA tile (alpha = mask). At every
training step a fresh batch of pairs is **rendered on the GPU** (`patsim/data.py: PairSynth`):

* partner: same photo / same breed / same species / other species (30/25/20/25 %) → pattern target
* geometry of B: identical to A / small jitter / independent (20/30/50 %), sizes 25–70 % of the image
* each image either a cut-out on a random procedural background, or inside its original photo
* shared random mirroring, brightness/contrast jitter, noise

Because the code does the shifting and resizing itself, the location and size labels are exact. The
network is trained with BCE on the pattern target, L1 on centres, smooth-L1 on log size, plus
auxiliary mask and breed losses. Evaluation uses a fixed set of pairs from the **test** split
(animals never seen in training).

## Run it

**Colab (recommended):** open `notebooks/colab_train.ipynb`, choose a T4 GPU runtime, set
`GITHUB_USER`, run all. Downloading and preprocessing take a few minutes, and 6,000 training steps take roughly 15–20 min.
Tiles and checkpoints are cached in Google Drive.

**Locally / CLI:**

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # 30 s CPU self-test, no download
python -m patsim.train --steps 6000 --out_dir runs/r18
```

**Use a trained model:**

```python
from patsim import PairScorer
scorer = PairScorer.from_checkpoint("runs/r18/patsim.pt")      # d0=..., tau=... to reshape scores
r = scorer.score("cat1.jpg", "cat2.jpg")                        # paths, PIL, numpy (H,W[,C]) or tensors
print(r["pattern"], r["location"], r["scale"], r["dx"], r["dy"], r["scale_ratio"])
```

Inputs of any size and aspect ratio are letterboxed to 160×160. Single-channel numpy
arrays (e.g. a 2D physical field) are min-max normalised and repeated to 3 channels.

## Putting it on GitHub

Without git installed: create an empty repo `pattern-sim` on github.com → *Add file → Upload files* →
drag in the contents of this folder → commit. With git:

```bash
git init -b main && git add . && git commit -m "pattern-sim" && git remote add origin https://github.com/YOUR_GITHUB_ID/pattern-sim.git && git push -u origin main
```

Keep the repo public, or the plain `git clone` in the notebook will need a token. You can also skip
GitHub and upload `pattern-sim.zip` to `/content` in Colab; the notebook unzips it automatically.

## Extending

* **Other objects:** any dataset with masks or boxes works. Write a loader that returns `make_tile` samples.
* **More scores:** add a label in `PairSynth.sample` (e.g. rotation angle or intensity scaling), a head
  in `PatSimNet.encode`, and a score in `scores.py`.
* **Scientific 2D fields:** the same pattern / location / scale split resembles object-based
  verification scores (e.g. SAL, MODE). To use it there, train on fields with known objects instead of pets.
* **Bigger backbone:** swap ResNet-18 for ResNet-50 in `model.py` (layer channel counts change to 1024/2048).

## Files

```
patsim/data.py      Oxford-IIIT Pet → tile bank; GPU pair renderer with exact labels
patsim/model.py     PatSimNet (Siamese ResNet-18 + mask / heat / size / embedding heads)
patsim/scores.py    score definitions (d0, tau)
patsim/train.py     training loop + CLI (Config dataclass)
patsim/evaluate.py  fixed-seed test metrics (MAE per score, centre error px, size error %, AUCs)
patsim/infer.py     PairScorer for arbitrary images
patsim/viz.py       plotting helpers
tests/              CPU smoke tests on fake data
notebooks/          Colab notebook
```
