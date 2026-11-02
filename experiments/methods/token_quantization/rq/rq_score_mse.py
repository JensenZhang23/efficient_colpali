#!/usr/bin/env python3
"""
RQ with query-aware weighted k-means (RQ-ScoreMSE).

Motivation
----------
Standard RQ minimises isotropic reconstruction MSE. But MaxSim only reads the
argmax token per query token, so tokens that are frequently selected as argmax
matter far more than padding-adjacent tokens. Weighted k-means biases centroids
towards minimising error at retrieval-critical positions:

    objective = sum_i w_i * ||x_i - c_{a_i}||^2

where w_i = 1 + BETA * argmax_count_i (BETA default 10).

Centroid update:  c_k = (sum_{i:a_i=k} w_i * x_i) / (sum_{i:a_i=k} w_i)

The weights are computed once from FP16 argmax positions and applied at every
RQ stage (residuals change but token importance doesn't).

Usage
-----
  CACHE_DIR=experiment/embeddings/colqwen2.5_v3/physics \
  ASCEND_RT_VISIBLE_DEVICES=8 \
  python3 tree_improvements/methods/rq/rq_score_mse.py

Env
---
  CACHE_DIR   path to embedding cache
  DEVICE      logical NPU (default npu:0)
  M_VAL       codebooks (default 16)
  K_VAL       codebook size (default 256)
  BETA        argmax weight multiplier (default 10)
  BQ          query batch for argmax scan (default 4)
"""
import os
import sys
import time
import json
from pathlib import Path


def _find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "experiment" / "__init__.py").is_file():
            return parent
    raise RuntimeError(f"experiment package root not found above {p}")


ROOT = _find_root()
sys.path.insert(0, str(ROOT))

import torch
import torch_npu  # noqa: F401

DEVICE = os.environ.get("DEVICE", "npu:0")
M_VAL  = int(os.environ.get("M_VAL", "16"))
K_VAL  = int(os.environ.get("K_VAL", "256"))
BETA   = float(os.environ.get("BETA", "10"))
BQ     = int(os.environ.get("BQ", "4"))
CACHE  = Path(os.environ.get(
    "CACHE_DIR",
    str(ROOT / "experiment/embeddings/colqwen2.5_v3/physics")))
OUT_DIR = ROOT / "experiment/results/rq_score_mse"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(*a): print(*a, flush=True)


def ndcg(sc, rel, k=10):
    k2 = min(k, sc.shape[1])
    order = sc.argsort(1, descending=True)[:, :k2]
    r = rel.float().gather(1, order)
    disc = 1.0 / torch.log2(torch.arange(1, k2 + 1, dtype=torch.float32) + 1)
    dcg = (r * disc).sum(1)
    idcg = (rel.float().sort(1, descending=True)[0][:, :k2] * disc).sum(1)
    m = idcg > 0
    return (dcg[m] / idcg[m]).mean().item()


@torch.no_grad()
def maxsim(query, page_flat, pmask, P, T, bq=8):
    Q, Tq, d = query.shape
    neg = torch.finfo(torch.float32).min
    mf = pmask.reshape(P, T).to(DEVICE)
    out = torch.zeros(Q, P)
    for qi in range(0, Q, bq):
        qb = query[qi:qi + bq].to(DEVICE)
        Bq = qb.shape[0]
        sim = (qb.reshape(Bq * Tq, d) @ page_flat.T).reshape(Bq, Tq, P, T)
        sim.masked_fill_(~mf.view(1, 1, P, T), neg)
        out[qi:qi + Bq] = sim.max(3)[0].sum(1).cpu()
    return out


@torch.no_grad()
def compute_argmax_counts(query, fp16_flat, mask_flat, P, T, bq=BQ):
    """
    For each valid token, count how many (query_token, page) pairs select it
    as argmax. Returns a (P*T,) float tensor; never-argmax tokens get 0.
    """
    Q, Tq, d = query.shape
    neg = torch.finfo(torch.float32).min
    mf = mask_flat.reshape(P, T).to(DEVICE)
    counts = torch.zeros(P * T, dtype=torch.float32)
    for qi in range(0, Q, bq):
        qb = query[qi:qi + bq].to(DEVICE)
        Bq = qb.shape[0]
        sim = (qb.reshape(Bq * Tq, d) @ fp16_flat.T).reshape(Bq * Tq, P, T)
        sim.masked_fill_(~mf.unsqueeze(0), neg)
        idx = sim.argmax(-1)  # (Bq*Tq, P)
        # convert (page, local_t) -> flat index
        page_off = torch.arange(P, device=DEVICE) * T  # (P,)
        flat_idx = (page_off.unsqueeze(0) + idx).reshape(-1).cpu()
        counts.index_add_(0, flat_idx, torch.ones(flat_idx.shape[0]))
    return counts  # (P*T,)


@torch.no_grad()
def _assign(x, c):
    """Nearest centroid assignment. Returns (N,) long."""
    out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    chunk = max(1, 2_000_000 // max(1, c.shape[0]))
    cnorm = (c * c).sum(1)
    for s in range(0, x.shape[0], chunk):
        xb = x[s:s + chunk]
        d2 = (xb * xb).sum(1, keepdim=True) - 2 * xb @ c.T + cnorm
        out[s:s + chunk] = d2.argmin(1)
    return out


@torch.no_grad()
def _weighted_kmeans(x, w, k, n_iter=25, seed=0):
    """
    Weighted euclidean k-means.
    x: (N, d), w: (N,) float weights, k: int
    Returns (k, d) centroids.
    """
    n, d = x.shape
    if k >= n:
        return x[:k].clone() if n >= k else torch.cat([x, x[:k - n]], 0).clone()
    g = torch.Generator(device="cpu").manual_seed(seed)
    centroids = x[torch.randperm(n, generator=g)[:k].to(x.device)].clone()
    prev_assign = None
    chunk = max(1, 2_000_000 // max(1, k))
    for _ in range(n_iter):
        assign = _assign(x, centroids)
        # weighted centroid update
        new  = torch.zeros_like(centroids)
        wsum = torch.zeros(k, device=x.device)
        for s in range(0, n, chunk):
            xe = x[s:s + chunk]
            we = w[s:s + chunk].to(x.device)
            ae = assign[s:s + chunk]
            new.index_add_(0, ae, xe * we.unsqueeze(1))
            wsum.index_add_(0, ae, we)
        empty = wsum == 0
        new = new / wsum.clamp_min(1e-12).unsqueeze(1)
        if empty.any():
            ne = int(empty.sum())
            new[empty] = x[torch.randperm(n, generator=g)[:ne].to(x.device)]
        centroids = new
        if prev_assign is not None and (assign != prev_assign).float().mean() < 1e-4:
            break
        prev_assign = assign
    return centroids


@torch.no_grad()
def fit_rq_weighted(x, w, num_codebooks, codebook_size, n_iter=25, seed=0):
    """
    Fit M residual codebooks with per-token weights.
    x: (N, d), w: (N,) — same weights applied at every stage.
    Returns list of (K, d) tensors.
    """
    residual = x.clone()
    codebooks = []
    for m in range(num_codebooks):
        c = _weighted_kmeans(residual, w, codebook_size, n_iter=n_iter, seed=seed + m)
        residual = residual - c[_assign(residual, c)]
        codebooks.append(c)
    return codebooks


def main():
    tag = f"M{M_VAL}_K{K_VAL}_beta{BETA}"
    log("=" * 70)
    log(f"RQ-ScoreMSE  {tag}  cache={CACHE.name}  device={DEVICE}")
    log(f"beta={BETA}  BQ={BQ}")
    log("=" * 70)

    # load embeddings
    page  = torch.load(CACHE / "page_embeddings.pt",  weights_only=True).float()
    query = torch.load(CACHE / "query_embeddings.pt", weights_only=True).float()
    rel   = torch.load(CACHE / "relevance.pt",        weights_only=True)
    P, T, d = page.shape
    Q = query.shape[0]
    pmask    = page.norm(dim=-1) > 1e-6
    flat_all = page.reshape(P * T, d)
    mask_all = pmask.reshape(P * T)
    valid    = flat_all[mask_all]
    N_valid  = valid.shape[0]
    log(f"P={P} T={T} d={d} Q={Q} valid={N_valid:,}")

    # FP16 baseline
    fp16_dev = flat_all.clone().to(DEVICE)
    fp16_dev[~mask_all.to(DEVICE)] = 0.0
    sc_fp16  = maxsim(query, fp16_dev, pmask, P, T)
    fp16_val = ndcg(sc_fp16, rel)
    log(f"FP16 nDCG@10 = {fp16_val:.4f}")

    # standard RQ baseline (unweighted)
    log("\nFitting standard RQ (unweighted)...")
    from experiments.methods.token_quantization.rq.rq_quant import fit_rq, encode_rq, reconstruct_rq
    t0 = time.time()
    cb_std = fit_rq(valid.to(DEVICE), M_VAL, K_VAL)
    codes_std = encode_rq(valid.to(DEVICE), cb_std)
    rec_std = reconstruct_rq(codes_std, cb_std).cpu()
    t_std = time.time() - t0
    rq_flat = torch.zeros(P * T, d)
    rq_flat[mask_all] = rec_std
    rq_flat[~mask_all] = 0.0
    sc_std   = maxsim(query, rq_flat.to(DEVICE), pmask, P, T)
    ndcg_std = ndcg(sc_std, rel)
    log(f"  standard RQ  nDCG = {ndcg_std:.4f}  ({ndcg_std/fp16_val*100:.2f}%)  fit={t_std:.1f}s")

    # compute argmax frequency weights
    log("\nComputing argmax frequency weights...")
    t0 = time.time()
    counts = compute_argmax_counts(query, fp16_dev, mask_all, P, T)  # (P*T,)
    t_cnt  = time.time() - t0
    # extract weights for valid tokens only
    counts_valid = counts[mask_all]     # (N_valid,)
    w_valid = 1.0 + BETA * counts_valid
    n_nonzero = (counts_valid > 0).sum().item()
    log(f"  argmax scan: {t_cnt:.1f}s  non-zero={n_nonzero:,}/{N_valid:,} "
        f"({n_nonzero/N_valid*100:.1f}%)  w range [{w_valid.min():.0f}, {w_valid.max():.0f}]")

    # weighted RQ
    log(f"\nFitting weighted RQ (beta={BETA})...")
    t0 = time.time()
    cb_w = fit_rq_weighted(valid.to(DEVICE), w_valid.to(DEVICE), M_VAL, K_VAL)
    # encode with nearest-centroid (no gradient, same as standard)
    from experiments.methods.token_quantization.rq.rq_quant import encode_rq, reconstruct_rq
    codes_w = encode_rq(valid.to(DEVICE), cb_w)
    rec_w   = reconstruct_rq(codes_w, cb_w).cpu()
    t_w = time.time() - t0
    rq_flat_w = torch.zeros(P * T, d)
    rq_flat_w[mask_all]  = rec_w
    rq_flat_w[~mask_all] = 0.0
    sc_w    = maxsim(query, rq_flat_w.to(DEVICE), pmask, P, T)
    ndcg_w  = ndcg(sc_w, rel)
    log(f"  weighted RQ  nDCG = {ndcg_w:.4f}  ({ndcg_w/fp16_val*100:.2f}%)  fit={t_w:.1f}s")

    delta = ndcg_w - ndcg_std
    log(f"\n  Delta vs standard RQ: {delta:+.4f}  ({delta/fp16_val*100:+.2f} pp)")
    if abs(delta) < 0.008:
        log("  (within ±0.008 NPU noise band — not conclusive)")

    result = dict(
        cache=CACHE.name, tag=tag, method="rq_score_mse",
        M=M_VAL, K=K_VAL, beta=BETA,
        ndcg_fp16=round(fp16_val, 4),
        ndcg_rq_std=round(ndcg_std, 4),
        ndcg_rq_weighted=round(ndcg_w, 4),
        quality_std_pct=round(ndcg_std / fp16_val * 100, 2),
        quality_w_pct=round(ndcg_w / fp16_val * 100, 2),
        delta=round(delta, 4),
        n_argmax_tokens=int(n_nonzero),
        argmax_coverage_pct=round(n_nonzero / N_valid * 100, 2),
        fit_time_std_s=round(t_std, 1),
        fit_time_w_s=round(t_w, 1),
    )
    out = OUT_DIR / f"rq_score_mse_{tag}_{CACHE.name}.json"
    out.write_text(json.dumps(result, indent=2))
    log(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
