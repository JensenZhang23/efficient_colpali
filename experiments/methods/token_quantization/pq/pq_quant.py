#!/usr/bin/env python3
"""Product Quantization (PQ) for ColPali token compression.

Split d into m_sub contiguous subvectors, fit a K-codebook on each, encode a
token as m_sub codebook indices -> m_sub * log2(K) / 8 bytes per token.

Same fit / encode / reconstruct pattern as RQ so it plugs into the shared
MaxSim/nDCG harness (reconstruct to FP16, then score).
"""
from __future__ import annotations

import math

import torch

from experiments.methods.token_quantization.rq.rq_quant import _assign, _euclidean_kmeans


@torch.no_grad()
def fit_pq(x, m_sub, K=256, n_iter=25, seed=0):
    """Split d into m_sub contiguous subvectors, fit a K-codebook on each.
    Returns list of m_sub codebooks, each (K, d/m_sub).
    """
    N, d = x.shape
    assert d % m_sub == 0, f"d={d} not divisible by m_sub={m_sub}"
    dsub = d // m_sub
    codebooks = []
    for i in range(m_sub):
        sub = x[:, i * dsub:(i + 1) * dsub].contiguous()
        c = _euclidean_kmeans(sub, K, n_iter=n_iter, seed=seed + i)
        codebooks.append(c)
    return codebooks


@torch.no_grad()
def encode_pq(x, codebooks):
    """Assign each subvector to its nearest centroid. Returns (N, m_sub) long."""
    dsub = codebooks[0].shape[1]
    codes = []
    for i, c in enumerate(codebooks):
        sub = x[:, i * dsub:(i + 1) * dsub].contiguous()
        codes.append(_assign(sub, c))
    return torch.stack(codes, dim=1)


@torch.no_grad()
def reconstruct_pq(codes, codebooks):
    """Concatenate per-subvector centroids. codes (N, m_sub) -> (N, d)."""
    parts = [c[codes[:, i]] for i, c in enumerate(codebooks)]
    return torch.cat(parts, dim=1)


def bytes_per_token_pq(m_sub, K=256):
    return m_sub * math.log2(K) / 8.0
