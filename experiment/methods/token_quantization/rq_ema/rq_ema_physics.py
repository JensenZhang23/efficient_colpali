#!/usr/bin/env python3
"""RQ-EMA vs static k-means RQ on ViDoRe physics, same 16 B/tok, nDCG@10.

EMA RQ trains codebooks VQ-VAE style: per layer, assign residuals to nearest
code, then update each code toward the EMA of its assigned residuals (no SGD).
A trainable-codebook baseline between static k-means RQ and learned RQ-MoE.
Reuses the fast NPU MaxSim+nDCG path so eval matches the RQ-MoE experiment.

Usage:
  DATASET=physics DEVICE=npu:0 EPOCHS=10 python3 tree_improvements/methods/rq_ema/rq_ema_physics.py
Env: M K EPOCHS DECAY
"""
import os
import sys
import time
import json
from pathlib import Path

def _find_root() -> Path:
    """Walk up to the directory containing the `experiment` package."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "experiment" / "__init__.py").is_file():
            return parent
    raise RuntimeError(f"experiment package root not found above {p}")


ROOT = _find_root()
sys.path.insert(0, str(ROOT))

import torch
import torch_npu  # noqa: F401

from experiment.methods.token_quantization.rq.rq_quant import fit_rq, encode_rq, reconstruct_rq
from experiment.methods.token_quantization.rq.rq_quant import _euclidean_kmeans, _assign

DEVICE = os.environ.get("DEVICE", "npu:0")
DATASET = os.environ.get("DATASET", "physics")
M = int(os.environ.get("M", "16"))
K = int(os.environ.get("K", "256"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
DECAY = float(os.environ.get("DECAY", "0.99"))

CACHE = ROOT / "experiment/embeddings/colqwen2_v1.0_v3" / DATASET
OUT = ROOT / "experiment/results/quantization_comparison" / f"rq_ema_{DATASET}.json"
OUT.parent.mkdir(exist_ok=True)


@torch.no_grad()
def score_fp16(query, page_flat, mask_flat, P, T, bq=8):
    Q, Tq, d = query.shape
    neg = torch.finfo(torch.float32).min
    scores = torch.zeros(Q, P)
    for qi in range(0, Q, bq):
        qb = query[qi:qi + bq].to(DEVICE)
        nb = qb.shape[0]
        sim = qb.reshape(nb * Tq, d) @ page_flat.T
        sim = sim.reshape(nb, Tq, P, T)
        sim.masked_fill_(~mask_flat.view(1, 1, P, T), neg)
        scores[qi:qi + nb] = sim.max(3).values.sum(1).cpu()
    return scores


def ndcg(scores, rel, k=10):
    k2 = min(k, scores.shape[1])
    order = scores.argsort(1, descending=True)[:, :k2]
    gains = rel.float().gather(1, order)
    disc = 1.0 / torch.log2(torch.arange(1, k2 + 1, dtype=torch.float32) + 1)
    dcg = (gains * disc).sum(1)
    idcg = (rel.float().sort(1, descending=True).values[:, :k2] * disc).sum(1)
    ok = idcg > 0
    return (dcg[ok] / idcg[ok]).mean().item()


@torch.no_grad()
def fit_rq_ema(x, M, K, n_epochs, decay, seed=0):
    """Vectorized EMA RQ. Returns list of M codebooks (K, d)."""
    torch.manual_seed(seed)
    d = x.shape[1]
    codebooks = []
    residual = x.clone()
    for m in range(M):
        c = _euclidean_kmeans(residual, k=K, n_iter=10)
        codebooks.append(c)
        idx = _assign(residual, c)
        residual = residual - c[idx]
    csize = [torch.ones(K, device=x.device) for _ in range(M)]
    csum = [c.clone() for c in codebooks]
    for epoch in range(n_epochs):
        residual = x.clone()
        for m in range(M):
            idx = _assign(residual, codebooks[m])
            n_k = torch.bincount(idx, minlength=K).float()
            sum_k = torch.zeros(K, d, device=x.device)
            sum_k.index_add_(0, idx, residual)
            csize[m] = decay * csize[m] + (1 - decay) * n_k
            csum[m] = decay * csum[m] + (1 - decay) * sum_k
            denom = csize[m].clamp_min(1e-6).unsqueeze(1)
            codebooks[m] = csum[m] / denom
            residual = residual - codebooks[m][idx]
        rr = x.clone()
        for m in range(M):
            idx = _assign(rr, codebooks[m])
            rr = rr - codebooks[m][idx]
        mse = (rr ** 2).sum(1).mean().item()
        print(f"epoch {epoch+1:3d}  recon MSE={mse:.4e}", flush=True)
    return codebooks


def main():
    print("=" * 70, flush=True)
    print(f"RQ-EMA vs static RQ  dataset={DATASET}  M={M} K={K} EPOCHS={EPOCHS} decay={DECAY} device={DEVICE}", flush=True)
    print("=" * 70, flush=True)

    page = torch.load(CACHE / "page_embeddings.pt", weights_only=True).float()
    query = torch.load(CACHE / "query_embeddings.pt", weights_only=True).float()
    rel = torch.load(CACHE / "relevance.pt", weights_only=True)
    P, T, d = page.shape
    Q = query.shape[0]
    page_mask = page.norm(dim=-1) > 1e-6
    flat = page.reshape(P * T, d)
    mask = page_mask.reshape(P * T)
    valid = flat[mask]
    print(f"Pages={P} T={T} d={d} Q={Q}  valid tokens={valid.shape[0]:,}", flush=True)

    pf_npu = flat.to(DEVICE); pf_npu[~mask.to(DEVICE)] = 0.0
    mf_npu = mask.to(DEVICE)
    valid_npu = valid.to(DEVICE)

    sc = score_fp16(query, pf_npu, mf_npu, P, T)
    ndcg_fp16 = ndcg(sc, rel)
    print(f"FP16 exact       nDCG@10={ndcg_fp16:.4f}", flush=True)

    def eval_cb(cb, tag):
        codes = encode_rq(valid_npu, cb)
        rec_v = reconstruct_rq(codes, cb)
        rec_flat = torch.zeros(P * T, d, device=DEVICE)
        rec_flat[mf_npu] = rec_v
        s = score_fp16(query, rec_flat, mf_npu, P, T)
        n = ndcg(s, rel)
        mse = ((rec_v - valid_npu) ** 2).sum(1).mean().item()
        print(f"{tag:18s} nDCG@10={n:.4f} ({n/ndcg_fp16*100:.2f}%)  MSE={mse:.4e}", flush=True)
        return n, mse

    print("Fitting static k-means RQ...", flush=True)
    t0 = time.time()
    cb_km = fit_rq(valid_npu, num_codebooks=M, codebook_size=K, n_iter=25, seed=0)
    t_km = time.time() - t0
    ndcg_km, mse_km = eval_cb(cb_km, f"k-means RQ M={M}")
    print(f"  k-means fit {t_km:.1f}s", flush=True)

    print(f"\nFitting EMA RQ ({EPOCHS} epochs)...", flush=True)
    t0 = time.time()
    cb_ema = fit_rq_ema(valid_npu, M, K, EPOCHS, DECAY, seed=0)
    t_ema = time.time() - t0
    ndcg_ema, mse_ema = eval_cb(cb_ema, f"EMA RQ M={M}")
    print(f"  EMA fit {t_ema:.1f}s ({t_ema/t_km:.2f}x k-means)", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"FP16 exact     nDCG@10={ndcg_fp16:.4f}  (100.00%)", flush=True)
    print(f"k-means RQ   nDCG@10={ndcg_km:.4f}  ({ndcg_km/ndcg_fp16*100:.2f}%)  MSE={mse_km:.4e}", flush=True)
    print(f"EMA RQ         nDCG@10={ndcg_ema:.4f}  ({ndcg_ema/ndcg_fp16*100:.2f}%)  MSE={mse_ema:.4e}", flush=True)
    print(f"delta nDCG (EMA - kmeans): {ndcg_ema - ndcg_km:+.4f}", flush=True)

    payload = dict(dataset=DATASET, device=DEVICE, M=M, K=K, epochs=EPOCHS, decay=DECAY,
        ndcg_fp16=round(ndcg_fp16, 4),
        ndcg_kmeans=round(ndcg_km, 4), mse_kmeans=mse_km,
        ndcg_ema=round(ndcg_ema, 4), mse_ema=mse_ema,
        delta_ndcg=round(ndcg_ema - ndcg_km, 4),
        t_kmeans=t_km, t_ema=t_ema,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
