"""Registration network: how is the forecast shifted / zoomed / rotated relative to the observation,
and how alike are the two fields once that is undone?"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .data import to_input
from .geometry import resample, sim_matrix, sim_params

PARAM_SCALE = (0.1, 0.1, 0.15, 0.35)  # typical size of (tx, ty, log_s, theta), used to normalise the loss
ENCODER_PREFIX = "encoder."


def _block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class RegNet(nn.Module):
    """
    register: shared ResNet-18 encoder (1-channel, stride 8) on A and B
              -> correlation of every A position with every B position
              -> CNN -> (tx, ty, log_s, theta) and an uncertainty for each      (Rocco et al. 2017 style)
    compare:  B moved back onto A with that transform; [A, B_aligned, A - B_aligned, valid] -> CNN
              -> pattern logit and log amount ratio (B / A)
    """

    def __init__(self, canvas=128, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            conv1.weight.copy_(r.conv1.weight.sum(1, keepdim=True))
        r.layer3[0].conv1.stride = (1, 1)  # keep stride 8 for a finer correlation grid
        r.layer3[0].downsample[0].stride = (1, 1)
        self.encoder = nn.Sequential(conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        h = canvas // 8
        self.regress = nn.Sequential(
            _block(h * h, 128), _block(128, 128, 2), _block(128, 64, 2), nn.Flatten(),
            nn.Linear(64 * math.ceil(h / 4) ** 2, 256), nn.ReLU(inplace=True), nn.Linear(256, 8))
        nn.init.zeros_(self.regress[-1].weight)  # start from "no transform"
        nn.init.zeros_(self.regress[-1].bias)
        self.compare_net = nn.Sequential(
            _block(4, 32, 2), _block(32, 32), _block(32, 64, 2), _block(64, 64), _block(64, 128, 2),
            _block(128, 128), _block(128, 256, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))
        self.register_buffer("param_scale", torch.tensor(PARAM_SCALE))

    def register(self, a, b):
        """Transform of b relative to a: params (n, 4) and log Laplace scales log_b (n, 4), both in
        physical units (tx, ty as fraction of the canvas; log_s; theta in radians)."""
        n = a.shape[0]
        f = self.encoder((to_input(torch.cat([a, b])) - 0.1) / 0.2)
        f = F.normalize(f.float(), dim=1)
        fa, fb = f[:n], f[n:]
        h, w = fa.shape[-2:]
        corr = torch.einsum("ncij,nckl->nklij", fa, fb).reshape(n, h * w, h, w)  # channels = B positions
        out = self.regress(F.normalize(F.relu(corr), dim=1)).float()
        return out[:, :4] * self.param_scale, out[:, 4:].clamp(-7, 3) + self.param_scale.log()

    def compare(self, a, b_aligned, valid):
        xa, xb = to_input(a), to_input(b_aligned)
        out = self.compare_net(torch.cat([xa, xb, xa - xb, valid], 1)).float()
        return out[:, 0], out[:, 1]

    def forward(self, a, b, refine=0):
        """a = observation, b = forecast, both (n, 1, L, L) in mm/h. `refine` extra passes estimate the
        remaining transform after aligning (useful at inference)."""
        params, log_b = self.register(a, b)
        M = sim_matrix(params.detach())
        for _ in range(refine):
            p2, log_b = self.register(a, resample(b, M))
            M = M @ sim_matrix(p2)
            params = sim_params(M)
        b_aligned, valid = resample(b, M), resample(torch.ones_like(b), M)  # b_aligned(q) = b(M q)
        pattern_logit, log_amp = self.compare(a, b_aligned, valid)
        return dict(params=params, log_b=log_b, pattern_logit=pattern_logit, log_amp=log_amp,
                    b_aligned=b_aligned, valid=valid)
