"""Training loop, saving/loading, and reuse of a saved model.

CLI:  python -m precipsim.train --data_path era5_ea.npz --steps 8000 --mode ask
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn.functional as F

from .data import FieldBank, FieldPairSynth
from .evaluate import evaluate
from .model import ENCODER_PREFIX, RegNet

CKPT_NAME = "precipsim.pt"
MODES = {"u": "use", "c": "continue", "s": "scratch", "a": "ask"}


@dataclass
class Config:
    data_path: str = "era5_ea.npz"      # FieldBank .npz (see data.download_era5 / FieldBank.from_array)
    out_dir: str = "runs/precip"        # checkpoint + history
    train_years: tuple = (2016, 2022)   # inclusive
    test_years: tuple = (2023, 2023)
    canvas: int = 128
    distortion: float = 1.0             # strength of forecast-like errors in the synthetic pairs
    steps: int = 8000                   # steps in this run (added on top when continuing)
    batch_size: int = 64                # pairs per step
    lr: float = 3e-4
    backbone_lr_mult: float = 0.5
    weight_decay: float = 1e-4
    warmup: int = 300
    amp: bool = True
    pretrained: bool = True             # ImageNet weights for a new model (ignored when continuing)
    seed: int = 0
    log_every: int = 100
    eval_every: int = 1000
    eval_pairs: int = 1024
    w_geo: float = 1.0
    w_unc: float = 0.1
    w_pattern: float = 1.0
    w_amp: float = 0.5


def compute_loss(out, batch, cfg, param_scale):
    """Transform + amount losses on 'same' pairs, uncertainty (Laplace NLL, estimate detached), pattern BCE."""
    same = batch["same"].float()
    n_same = same.sum().clamp_min(1)
    err = (out["params"] - batch["params"]) / param_scale
    geo = F.smooth_l1_loss(err, torch.zeros_like(err), beta=0.1, reduction="none").mean(1)
    abs_err = (out["params"].detach() - batch["params"]).abs()
    unc = (out["log_b"] + abs_err / out["log_b"].exp()).mean(1)
    amp_err = (out["log_amp"] - batch["log_a"]) / 0.35
    amp = F.smooth_l1_loss(amp_err, torch.zeros_like(amp_err), beta=0.1, reduction="none")
    terms = dict(geo=(geo * same).sum() / n_same, unc=(unc * same).sum() / n_same,
                 pattern=F.binary_cross_entropy_with_logits(out["pattern_logit"], batch["pattern"]),
                 amp=(amp * same).sum() / n_same)
    total = sum(getattr(cfg, f"w_{k}") * v for k, v in terms.items())
    return total, {k: float(v.detach()) for k, v in terms.items()}


def checkpoint_path(cfg):
    return os.path.join(cfg.out_dir, CKPT_NAME)


def save_checkpoint(path, model, cfg, history):
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "history": history,
                "saved_at": time.strftime("%Y-%m-%d %H:%M")}, path)
    with open(os.path.splitext(path)[0] + "_history.json", "w") as f:
        json.dump(history, f, indent=1)


def load_model(path, device=None):
    """(RegNet with the saved weights, checkpoint dict)."""
    ck = torch.load(path, map_location="cpu")
    model = RegNet(canvas=ck["config"]["canvas"], pretrained=False)
    model.load_state_dict(ck["model"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return model.to(device).eval(), ck


def describe_checkpoint(path):
    """A short summary of the saved model at `path`, or None if there is none."""
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location="cpu")
    last = (ck.get("history") or [{}])[-1]
    lines = [f"Saved model: {path}", f"  saved {ck.get('saved_at', '?')} after {last.get('step', '?')} training steps"]
    if "shift_err_px" in last:
        lines.append(f"  test errors: shift {last['shift_err_px']:.2f} grid pts, size {last['scale_err_pct']:.1f} %, "
                     f"rotation {last['rot_err_deg']:.1f} deg, amount {last['amp_err_pct']:.1f} %, "
                     f"pattern MAE {last['pattern_mae']:.3f}")
    return "\n".join(lines)


def snapshot_checkpoint(path):
    """Copy the checkpoint to a dated file next to it, which later runs never overwrite; returns its path."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(path)))
    dst = os.path.join(os.path.dirname(path), f"{os.path.splitext(CKPT_NAME)[0]}_{stamp}.pt")
    if not os.path.exists(dst):
        shutil.copy2(path, dst)
    return dst


def _ask():
    while True:
        answer = MODES.get(input("Use it as is [u], continue training from it [c], or train from scratch [s]? ")
                           .strip().lower()[:1])
        if answer in ("use", "continue", "scratch"):
            return answer
        print("Please type u, c or s.")


def prepare_model(cfg: Config, train_bank=None, test_bank=None, mode="ask", device=None):
    """(model, history), reusing the model saved in cfg.out_dir if there is one.

    mode: "use" (load it, no training), "continue" (train cfg.steps more from its weights),
    "scratch" (train a new model), or "ask" (prompt when a saved model exists, else train from
    scratch). Before "continue" or "scratch" overwrite the saved model, a dated copy is kept.
    """
    mode = MODES.get(str(mode).strip().lower()[:1])
    if mode is None:
        raise ValueError('mode must be "ask", "use", "continue" or "scratch"')
    path = checkpoint_path(cfg)
    info = describe_checkpoint(path)
    if info is None:
        if mode in ("use", "continue"):
            print(f"No saved model at {path}: training from scratch.")
        mode = "scratch"
    else:
        print(info)
        if mode == "ask":
            mode = _ask()
    if mode == "use":
        model, ck = load_model(path, device)
        print("Using the saved model.")
        return model, ck.get("history", [])
    if info is not None:
        print(f"Kept a copy of the saved model: {snapshot_checkpoint(path)}")
    return train(cfg, train_bank, test_bank, device, init_from=path if mode == "continue" else None)


def train(cfg: Config, train_bank=None, test_bank=None, device=None, init_from=None):
    """Train RegNet, from scratch or from the weights in checkpoint `init_from`; returns (model, history).
    Banks default to cfg.data_path split by year. The checkpoint is saved at every evaluation."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    if train_bank is None or test_bank is None:
        bank = FieldBank.load(cfg.data_path)
        train_bank = train_bank or bank.select_years(*cfg.train_years)
        test_bank = test_bank or bank.select_years(*cfg.test_years)
    train_synth = FieldPairSynth(train_bank.to(device), canvas=cfg.canvas, distortion=cfg.distortion)
    test_synth = FieldPairSynth(test_bank.to(device), canvas=cfg.canvas, distortion=cfg.distortion)

    model = RegNet(canvas=cfg.canvas, pretrained=cfg.pretrained and init_from is None)
    history, offset = [], 0
    if init_from:
        ck = torch.load(init_from, map_location="cpu")
        if ck["config"]["canvas"] != cfg.canvas:
            raise ValueError(f"saved model uses canvas {ck['config']['canvas']}, config has {cfg.canvas}")
        model.load_state_dict(ck["model"])
        history = list(ck.get("history", []))
        offset = history[-1]["step"] if history else 0
        print(f"Continuing from {init_from} ({offset} steps so far).")
    model = model.to(device)

    backbone = [p for n, p in model.named_parameters() if n.startswith(ENCODER_PREFIX)]
    heads = [p for n, p in model.named_parameters() if not n.startswith(ENCODER_PREFIX)]
    opt = torch.optim.AdamW([{"params": backbone, "lr": cfg.lr * cfg.backbone_lr_mult},
                             {"params": heads, "lr": cfg.lr}], weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, cfg.steps) / cfg.steps)))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(cfg.seed + offset)
    ckpt = checkpoint_path(cfg)
    t0 = time.time()

    model.train()
    for step in range(1, cfg.steps + 1):
        with torch.no_grad():
            batch = train_synth.sample(cfg.batch_size, rng)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out = model(batch["a"], batch["b"])
        loss, parts = compute_loss(out, batch, cfg, model.param_scale)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % cfg.log_every == 0:
            print(f"step {offset + step:5d}  loss {float(loss.detach()):.3f}  "
                  + "  ".join(f"{k} {v:.3f}" for k, v in parts.items()) + f"  ({time.time() - t0:.0f}s)")
        if step % cfg.eval_every == 0 or step == cfg.steps:
            metrics = evaluate(model, test_synth, cfg.eval_pairs)
            history.append({"step": offset + step, "loss": float(loss.detach()), **metrics})
            print("  eval  " + "  ".join(f"{k} {v:.3f}" for k, v in metrics.items()))
            save_checkpoint(ckpt, model, cfg, history)
    print(f"Saved model: {ckpt}")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train the precipitation registration / similarity network.")
    parse = {bool: lambda s: s.lower() in ("1", "true", "yes"),
             tuple: lambda s: tuple(int(x) for x in s.split(","))}  # e.g. --train_years 2016,2022
    for f in fields(Config):
        kind = type(f.default)
        parser.add_argument(f"--{f.name}", default=f.default, type=parse.get(kind, kind))
    parser.add_argument("--mode", default="ask", help="ask / use / continue / scratch when a saved model exists")
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    prepare_model(Config(**args), mode=mode)


if __name__ == "__main__":
    main()
