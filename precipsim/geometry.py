"""Similarity transforms on a square canvas, in grid_sample's normalised coordinates.

Canvas coordinates q = (x, y) lie in [-1, 1]: x points right (east on a north-up map), y points
down (along rows). A transform with parameters

    tx, ty   shift, as a fraction of the canvas side (2 * t in normalised units)
    log_s    log zoom factor
    theta    rotation in radians; positive = clockwise on screen (because y points down)

moves the content at p to   M p = s R(theta) p + 2 t   (zoom and rotation about the canvas centre).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

PARAMS = ("tx", "ty", "log_s", "theta")


def sim_matrix(p):
    """(n, 4) params -> (n, 3, 3) matrices M."""
    tx, ty, log_s, theta = p.float().unbind(-1)
    s = log_s.exp()
    M = torch.zeros(p.shape[0], 3, 3, device=p.device)
    M[:, 0, 0], M[:, 0, 1], M[:, 0, 2] = s * theta.cos(), -s * theta.sin(), 2 * tx
    M[:, 1, 0], M[:, 1, 1], M[:, 1, 2] = s * theta.sin(), s * theta.cos(), 2 * ty
    M[:, 2, 2] = 1
    return M


def sim_params(M):
    """(n, 3, 3) similarity matrices -> (n, 4) params."""
    a, b = M[:, 0, 0], M[:, 1, 0]
    return torch.stack([M[:, 0, 2] / 2, M[:, 1, 2] / 2, 0.5 * torch.log(a * a + b * b), torch.atan2(b, a)], -1)


def resample(img, P):
    """out(q) = img(P q) for img (n, C, L, L) and P (n, 3, 3); points outside the image read 0."""
    grid = F.affine_grid(P[:, :2].float(), list(img.shape), align_corners=False)
    return F.grid_sample(img.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def warp(img, M):
    """Move the content of img by M: out(M p) = img(p)."""
    return resample(img, torch.linalg.inv(M.float()))
