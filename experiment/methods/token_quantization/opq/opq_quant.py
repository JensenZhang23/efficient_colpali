#!/usr/bin/env python3
"""Optimized Product Quantization (OPQ) for ColPali token compression.

PQ plus a learned orthogonal rotation R that decorrelates dimensions before
splitting them into subvectors, then aligns the reconstruction back:

  (1) fit PQ on y = x @ R
  (2) Procrustes-solve R to align x with the PQ reconstruction

Same fit / encode / reconstruct pattern as RQ so it plugs into the shared
MaxSim/nDCG harness (reconstruct to FP16, then score).
"""
from __future__ import annotations

import torch

from experiment.methods.token_quantization.pq.pq_quant import encode_pq, fit_pq, reconstruct_pq


@torch.no_grad()
def fit_opq(x, m_sub, K=256, n_iter_km=25, n_iter_opq=10, seed=0):
    """Learn orthogonal rotation R (d,d) + PQ codebooks on rotated data.

    Returns (R, codebooks) where codebooks live in the rotated space.
    """
    N, d = x.shape
    # init rotation with PCA (decorrelates dims)
    xc = x - x.mean(0, keepdim=True)
    cov = (xc.T @ xc) / max(1, N - 1)
    _, evecs = torch.linalg.eigh(cov)
    R = evecs  # (d, d) orthogonal

    codebooks = None
    for _ in range(n_iter_opq):
        y = x @ R
        codebooks = fit_pq(y, m_sub, K, n_iter=n_iter_km, seed=seed)
        codes = encode_pq(y, codebooks)
        y_hat = reconstruct_pq(codes, codebooks)  # (N, d)
        # Procrustes: min_R ||x @ R - y_hat||_F, R orthogonal
        M = x.T @ y_hat  # (d, d)
        U, _, Vt = torch.linalg.svd(M)
        R = U @ Vt
    return R, codebooks


@torch.no_grad()
def encode_opq(x, R, codebooks):
    return encode_pq(x @ R, codebooks)


@torch.no_grad()
def reconstruct_opq(codes, R, codebooks):
    """Reconstruct in original space: y_hat @ R^T."""
    y_hat = reconstruct_pq(codes, codebooks)
    return y_hat @ R.T
