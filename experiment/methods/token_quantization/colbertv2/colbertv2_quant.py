#!/usr/bin/env python3
"""ColBERTv2-style residual compression for ColPali token compression.

Global K-centroid codebook plus per-dimension residual scalar buckets: a token
costs log2(K)/8 bytes for the centroid id + d * bits / 8 bytes for the residual.

Same fit / encode / reconstruct pattern as RQ so it plugs into the shared
MaxSim/nDCG harness (reconstruct to FP16, then score).
"""
from __future__ import annotations

import math

import torch

from experiment.methods.token_quantization.rq.rq_quant import _assign, _euclidean_kmeans


@torch.no_grad()
def fit_colbertv2(x, K=256, bits=2, n_iter=25, seed=0):
    """Fit K centroids; derive global per-dim residual quantization buckets.

    Returns dict with keys: centroids, cutoffs, bvals, bits.
    """
    centroids = _euclidean_kmeans(x, K, n_iter=n_iter, seed=seed)  # (K, d)
    assign = _assign(x, centroids)
    residual = x - centroids[assign]  # (N, d)

    # global scalar buckets over all residual values (ColBERTv2 style)
    flat = residual.reshape(-1)
    n_buckets = 2 ** bits
    qs = torch.linspace(0, 1, n_buckets + 1, device=x.device)[1:-1]
    cutoffs = torch.quantile(flat[:min(flat.numel(), 5_000_000)], qs)
    idx = torch.bucketize(flat, cutoffs)  # (N*d,) in [0, n_buckets-1]
    bvals = torch.zeros(n_buckets, device=x.device)
    for b in range(n_buckets):
        sel = idx == b
        bvals[b] = flat[sel].mean() if sel.any() else 0.0
    return {"centroids": centroids, "cutoffs": cutoffs, "bvals": bvals, "bits": bits}


@torch.no_grad()
def encode_colbertv2(x, model):
    centroids = model["centroids"]
    assign = _assign(x, centroids)  # (N,)
    residual = x - centroids[assign]
    codes = torch.bucketize(residual, model["cutoffs"])  # (N, d)
    return assign, codes


@torch.no_grad()
def reconstruct_colbertv2(assign, codes, model):
    return model["centroids"][assign] + model["bvals"][codes]  # (N, d)


def bytes_per_token_colbertv2(K=256, bits=2, d=128):
    return math.log2(K) / 8.0 + d * bits / 8.0
