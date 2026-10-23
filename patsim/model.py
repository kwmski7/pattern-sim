"""Siamese, object-centric network: one pass per image, then the pair is compared."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

BACKBONE_PREFIXES = ("stem.", "layer3.", "layer4.")


def _conv_head(dim, hidden=128):
    return nn.Sequential(nn.Conv2d(dim, hidden, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(hidden, 1, 1))


class PatSimNet(nn.Module):
    """Per image (shared weights):
        ResNet-18      -> stride-16 feature map (layer3 + upsampled layer4)
        mask head      -> soft object mask, used to pool features over the object only
        heat head      -> soft-argmax -> object centre (x, y) in [0, 1]
        size head      -> log(object size / image size)
        embedding      -> unit vector from mask-pooled features, blind to where/how big the object is
    Per pair:
        pattern logit  = a * cos(emb_A, emb_B) + b; location/scale come from the two centres/sizes
                         (see scores.pair_scores).
    """

    def __init__(self, n_classes=37, emb_dim=128, feat_dim=256, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2)
        self.layer3, self.layer4 = r.layer3, r.layer4
        self.lat3, self.lat4 = nn.Conv2d(256, feat_dim, 1), nn.Conv2d(512, feat_dim, 1)
        self.fuse = nn.Sequential(nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True))
        self.mask_head, self.heat_head = _conv_head(feat_dim), _conv_head(feat_dim)
        pooled = feat_dim + 512
        self.size_head = nn.Sequential(nn.Linear(pooled + 3, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))
        nn.init.constant_(self.size_head[-1].bias, math.log(0.4))
        self.embed = nn.Sequential(nn.Linear(pooled, 512), nn.ReLU(inplace=True), nn.Linear(512, emb_dim))
        self.classifier = nn.Linear(emb_dim, n_classes)  # auxiliary breed loss shapes the embedding
        self.sim_scale = nn.Parameter(torch.tensor(10.0))
        self.sim_bias = nn.Parameter(torch.tensor(-5.0))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def encode(self, x):
        """x: (n, 3, H, W) in [0, 1] -> dict of per-image outputs."""
        c3 = self.layer3(self.stem((x - self.mean) / self.std))
        c4 = F.interpolate(self.layer4(c3), size=c3.shape[-2:], mode="bilinear", align_corners=False)
        f = self.fuse(self.lat3(c3) + self.lat4(c4))
        n, _, h, w = f.shape
        xs = (torch.arange(w, device=x.device, dtype=torch.float32) + 0.5) / w
        ys = (torch.arange(h, device=x.device, dtype=torch.float32) + 0.5) / h

        mask_logit = self.mask_head(f)
        wts = torch.sigmoid(mask_logit)
        wts = wts / (wts.sum(dim=(2, 3), keepdim=True) + 1e-6)
        pooled = torch.cat([(f * wts).sum((2, 3)), (c4 * wts).sum((2, 3))], 1)

        heat = self.heat_head(f)
        p = heat.flatten(1).float().softmax(1).view(n, h, w)
        center = torch.stack([(p.sum(1) * xs).sum(1), (p.sum(2) * ys).sum(1)], 1)

        # Coverage and spread of the mask: cheap geometric cues for the size head.
        wm = wts.squeeze(1).float()
        wx, wy = wm.sum(1), wm.sum(2)
        mx, my = (wx * xs).sum(1, keepdim=True), (wy * ys).sum(1, keepdim=True)
        spread = torch.stack([torch.sigmoid(mask_logit).float().mean((1, 2, 3)),
                              ((wx * (xs - mx) ** 2).sum(1) + 1e-6).sqrt(),
                              ((wy * (ys - my) ** 2).sum(1) + 1e-6).sqrt()], 1)
        log_size = self.size_head(torch.cat([pooled, spread], 1)).squeeze(1)

        z = self.embed(pooled)
        return dict(emb=F.normalize(z.float(), dim=1), center=center, log_size=log_size.float(),
                    mask_logit=mask_logit, heat=heat, cls_logit=self.classifier(z))

    def forward(self, img_a, img_b):
        n = img_a.shape[0]
        out = self.encode(torch.cat([img_a, img_b]))
        a = {k: v[:n] for k, v in out.items()}
        b = {k: v[n:] for k, v in out.items()}
        cos = (a["emb"] * b["emb"]).sum(1)
        return dict(a=a, b=b, pattern_logit=self.sim_scale * cos + self.sim_bias)
