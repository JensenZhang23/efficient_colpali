#!/usr/bin/env python3
"""Residual (Vector) Quantization for ColBERT-style visual token compression.

Unlike tree/local_merge/PTM which reduce the NUMBER of tokens per page, RQ
reduces the BYTES per token: each 128-d vector is approximated by a sum of M
codewords, one from each of M learned codebooks of size K. Stored as M integer
codes (M*log2(K) bits); at scoring time we reconstruct x_hat = sum_m C_m[code_m]
back to float, so RQ plugs into the existing MaxSim/nDCG harness unchanged
(token count preserved). Reconstruction-based cousin of ColBERTv2/PLAID residual
compression, but codebooks are fit jointly across residual stages on the actual
visual-document token distribution. Euclidean k-means (not the cosine kmeans_fit
in colbertsar.py) since residuals are not unit-norm and reconstruction is raw.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def _euclidean_kmeans(x, k, n_iter=25, seed=0, tol=1e-4):
    n, d = x.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    if k >= n:
        idx = torch.randint(n, (k - n,), generator=g, device="cpu").to(x.device) if k > n else torch.empty(0, dtype=torch.long, device=x.device)
        pad = x[idx] if k > n else x[:0]
        return torch.cat([x, pad], dim=0)[:k].clone()
    centroids = x[torch.randperm(n, generator=g)[:k].to(x.device)].clone()
    prev_assign = None
    chunk = max(1, 2_000_000 // max(1, k))
    for _ in range(n_iter):
        assign = torch.empty(n, dtype=torch.long, device=x.device)
        cnorm = (centroids * centroids).sum(1)
        for s in range(0, n, chunk):
            xb = x[s:s + chunk]
            d2 = (xb * xb).sum(1, keepdim=True) - 2 * xb @ centroids.T + cnorm
            assign[s:s + chunk] = d2.argmin(1)
        new = torch.zeros_like(centroids)
        cnt = torch.zeros(k, device=x.device)
        new.index_add_(0, assign, x)
        cnt.index_add_(0, assign, torch.ones(n, device=x.device))
        empty = cnt == 0
        new = new / cnt.clamp_min(1.0).unsqueeze(1)
        if empty.any():
            ne = int(empty.sum())
            new[empty] = x[torch.randperm(n, generator=g)[:ne].to(x.device)]
        centroids = new
        if prev_assign is not None and (assign != prev_assign).float().mean() < tol:
            break
        prev_assign = assign
    return centroids


@torch.no_grad()
def _assign(x, c):
    n = x.shape[0]
    out = torch.empty(n, dtype=torch.long, device=x.device)
    chunk = max(1, 2_000_000 // max(1, c.shape[0]))
    cnorm = (c * c).sum(1)
    for s in range(0, n, chunk):
        xb = x[s:s + chunk]
        d2 = (xb * xb).sum(1, keepdim=True) - 2 * xb @ c.T + cnorm
        out[s:s + chunk] = d2.argmin(1)
    return out


@torch.no_grad()
def fit_rq(x, num_codebooks, codebook_size, n_iter=25, seed=0, sample_frac=0.1):
    """Fit M residual codebooks on rows of x. Returns list of (K, d) tensors.

    To ensure determinism, k-means runs on CPU (NPU matmul is non-deterministic).
    Training uses sample_frac of the data (default 10%, standard for large-scale VQ).
    Only the final codebooks are returned on the input device.
    """
    orig_device = x.device
    n = x.shape[0]
    n_train = max(codebook_size * 100, int(n * sample_frac))  # at least 100 points per centroid

    # Deterministic sampling
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:n_train]
    residual = x[idx].cpu()

    codebooks = []
    for m in range(num_codebooks):
        c = _euclidean_kmeans(residual, codebook_size, n_iter=n_iter, seed=seed + m)
        residual = residual - c[_assign(residual, c)]
        codebooks.append(c.to(orig_device))  # return on original device
    return codebooks


@torch.no_grad()
def encode_rq(x, codebooks):
    """Greedy residual encoding. Returns codes (N, M) long."""
    residual = x.clone()
    codes = []
    for c in codebooks:
        a = _assign(residual, c)
        residual = residual - c[a]
        codes.append(a)
    return torch.stack(codes, dim=1)


@torch.no_grad()
def reconstruct_rq(codes, codebooks):
    """Reconstruct x_hat = sum_m C_m[code_m]. codes (N, M) -> (N, d)."""
    out = torch.zeros(codes.shape[0], codebooks[0].shape[1], device=codes.device)
    for m, c in enumerate(codebooks):
        out += c[codes[:, m]]
    return out


def bytes_per_token(num_codebooks, codebook_size):
    """Storage cost of the codes for one token (codebooks amortized)."""
    return num_codebooks * math.log2(codebook_size) / 8.0


@torch.no_grad()
def quantize_pages(page, codebooks, device):
    """Quantize+reconstruct every non-pad token of a [P, T, d] page tensor.

    Padding (zero-norm rows) preserved exactly so avg_tok/masking is unchanged.
    """
    out = torch.zeros_like(page)
    for pi in range(page.shape[0]):
        pg = page[pi].to(device)
        mask = pg.norm(dim=-1) > 1e-6
        if mask.any():
            rec = reconstruct_rq(encode_rq(pg[mask], codebooks), codebooks)
            pg = pg.clone()
            pg[mask] = rec
        out[pi] = pg.cpu()
    return out
