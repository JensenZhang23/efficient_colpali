#!/usr/bin/env python3
"""Binary (1-bit sign) quantization for ColPali token compression.

Keeps only the sign of each dimension, with a learned per-dimension scale:
d / 8 bytes per token. The floor of the compression ladder.

Same fit / encode / reconstruct pattern as RQ so it plugs into the shared
MaxSim/nDCG harness (reconstruct to FP16, then score).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def fit_binary(x):
    """Learn per-dim scale = mean absolute value."""
    return x.abs().mean(0)  # (d,)


@torch.no_grad()
def encode_binary(x):
    return x > 0  # (N, d) bool


@torch.no_grad()
def reconstruct_binary(bits, scale):
    return (bits.float() * 2.0 - 1.0) * scale  # {-1, +1} * scale


def bytes_per_token_binary(d=128):
    return d / 8.0
