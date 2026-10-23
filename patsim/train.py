"""Training loop.   CLI:  python -m patsim.train --steps 6000 --out_dir runs/r18"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn.functional as F

from .data import PairSynth, build_bank
from .evaluate import evaluate
from .model import BACKBONE_PREFIXES, PatSimNet


@dataclass
class Config:
    data_root: str = "data"        # Oxford-IIIT Pet is downloaded here
    cache_dir: str = "cache"       # preprocessed tile banks
    out_dir: str = "runs/patsim"   # checkpoint + history
    tile: int = 160
    canvas: int = 160
    p_photo: float = 0.5           # share of images shown inside their original photo (vs. cut out)
    size_range: tuple = (0.25, 0.7)                 # object size, fraction of image side
    pattern_levels: tuple = (1.0, 0.75, 0.4, 0.0)   # same animal / breed / species / other species
    steps: int = 6000
    batch_size: int = 64           # pairs per step
    lr: float = 3e-4
    backbone_lr_mult: float = 0.3
    weight_decay: float = 1e-4
    warmup: int = 300
    amp: bool = True
    pretrained: bool = True
    seed: int = 0
    log_every: int = 100
    eval_every: int = 1000
    eval_pairs: int = 1024
    w_pattern: float = 1.0
    w_center: float = 10.0
    w_size: float = 2.0
    w_mask: float = 0.5
    w_cls: float = 0.1


def compute_loss(out, batch, cfg):
    """Pair loss (pattern) + per-image losses (centre, size, mask, breed)."""
    terms = dict(pattern=F.binary_cross_entropy_with_logits(out["pattern_logit"].float(), batch["pattern"]),
                 center=0.0, size=0.0, mask=0.0, cls=0.0)
    for s in "ab":
        o = out[s]
        target_mask = F.adaptive_avg_pool2d(batch[f"mask_{s}"], o["mask_logit"].shape[-2:])
        terms["center"] += F.l1_loss(o["center"], batch[f"center_{s}"]) / 2
        terms["size"] += F.smooth_l1_loss(o["log_size"], batch[f"size_{s}"].log(), beta=0.05) / 2
        terms["mask"] += F.binary_cross_entropy_with_logits(o["mask_logit"].float(), target_mask) / 2
        terms["cls"] += F.cross_entropy(o["cls_logit"].float(), batch[f"breed_{s}"]) / 2
    total = sum(getattr(cfg, f"w_{k}") * v for k, v in terms.items())
    return total, {k: float(v) for k, v in terms.items()}


def save_checkpoint(path, model, cfg, classes, history):
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "classes": list(classes),
                "history": history}, path)
    with open(os.path.splitext(path)[0] + "_history.json", "w") as f:
        json.dump(history, f, indent=1)


def train(cfg: Config, train_bank=None, test_bank=None, device=None):
    """Train PatSimNet; returns (model, history). Banks are built from Oxford-IIIT Pet if not given."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    if train_bank is None:
        train_bank = build_bank(cfg.data_root, "trainval", cfg.tile, cfg.cache_dir)
    if test_bank is None:
        test_bank = build_bank(cfg.data_root, "test", cfg.tile, cfg.cache_dir)
    train_bank, test_bank = train_bank.to(device), test_bank.to(device)
    synth_kw = dict(canvas=cfg.canvas, p_photo=cfg.p_photo, size_range=cfg.size_range,
                    pattern_levels=cfg.pattern_levels)
    train_synth, test_synth = PairSynth(train_bank, **synth_kw), PairSynth(test_bank, **synth_kw)
    model = PatSimNet(n_classes=len(train_bank.classes), pretrained=cfg.pretrained).to(device)

    backbone = [p for n, p in model.named_parameters() if n.startswith(BACKBONE_PREFIXES)]
    heads = [p for n, p in model.named_parameters() if not n.startswith(BACKBONE_PREFIXES)]
    opt = torch.optim.AdamW([{"params": backbone, "lr": cfg.lr * cfg.backbone_lr_mult},
                             {"params": heads, "lr": cfg.lr}], weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, cfg.steps) / cfg.steps)))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(cfg.seed)
    ckpt = os.path.join(cfg.out_dir, "patsim.pt")
    history, t0 = [], time.time()

    model.train()
    for step in range(1, cfg.steps + 1):
        with torch.no_grad():
            batch = train_synth.sample(cfg.batch_size, rng)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out = model(batch["img_a"], batch["img_b"])
        loss, parts = compute_loss(out, batch, cfg)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % cfg.log_every == 0:
            print(f"step {step:5d}  loss {float(loss):.3f}  "
                  + "  ".join(f"{k} {v:.3f}" for k, v in parts.items()) + f"  ({time.time() - t0:.0f}s)")
        if step % cfg.eval_every == 0 or step == cfg.steps:
            metrics = evaluate(model, test_synth, cfg.eval_pairs)
            history.append({"step": step, "loss": float(loss), **metrics})
            print("  eval  " + "  ".join(f"{k} {v:.3f}" for k, v in metrics.items()))
            save_checkpoint(ckpt, model, cfg, train_bank.classes, history)
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train the pattern / location / scale similarity network.")
    parse = {bool: lambda s: s.lower() in ("1", "true", "yes"),
             tuple: lambda s: tuple(float(x) for x in s.split(","))}  # e.g. --pattern_levels 1,0.8,0.5,0
    for f in fields(Config):
        kind = type(f.default)
        parser.add_argument(f"--{f.name}", default=f.default, type=parse.get(kind, kind))
    train(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
