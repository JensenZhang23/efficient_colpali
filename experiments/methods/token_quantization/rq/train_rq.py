#!/usr/bin/env python3
"""RQ M=16 with deterministic CPU k-means."""
import os
import sys
import json
import time
from pathlib import Path

def _find_root():
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "experiment" / "__init__.py").is_file():
            return parent
    raise RuntimeError(f"experiment package root not found above {p}")

ROOT = _find_root()
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch_npu

# 确定性设置
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
torch.manual_seed(42)
torch.npu.manual_seed_all(42)

from experiments.methods.token_quantization.rq.rq_quant import fit_rq, encode_rq, reconstruct_rq

import argparse as _ap
_p = _ap.ArgumentParser()
_p.add_argument("--cache-dir", type=Path,
    default=ROOT / "experiment/embeddings/colqwen2.5_v3/physics")
_p.add_argument("--out-dir", type=Path,
    default=ROOT / "experiment/results/quant")
_p.add_argument("--device", default="npu:0")
_p.add_argument("--m", type=int, default=16)
_p.add_argument("--k", type=int, default=256)
_args = _p.parse_args()

DEVICE = _args.device
M_VAL  = _args.m
K_VAL  = _args.k
CACHE  = _args.cache_dir
OUT    = _args.out_dir
OUT.mkdir(parents=True, exist_ok=True)

def log(*a): print(*a, flush=True)
def sync(): torch.npu.synchronize()

def ndcg(sc, rel, k=10):
    k2 = min(k, sc.shape[1])
    order = sc.argsort(1, descending=True)[:, :k2]
    r = rel.float().gather(1, order)
    disc = 1.0 / torch.log2(torch.arange(1, k2+1, dtype=torch.float32) + 1)
    dcg = (r * disc).sum(1)
    idcg = (rel.float().sort(1, descending=True)[0][:, :k2] * disc).sum(1)
    m = idcg > 0
    return (dcg[m] / idcg[m]).mean().item()

@torch.no_grad()
def maxsim(query, page_flat, pmask_flat, P, T, bq=8):
    Q, Tq, d = query.shape
    neg = torch.finfo(torch.float32).min
    mf = pmask_flat.reshape(P, T).to(DEVICE)
    out = torch.zeros(Q, P)
    for qi in range(0, Q, bq):
        qb = query[qi:qi+bq].to(DEVICE)
        Bq = qb.shape[0]
        sim = (qb.reshape(Bq*Tq, d) @ page_flat.T).reshape(Bq, Tq, P, T)
        sim.masked_fill_(~mf.view(1, 1, P, T), neg)
        out[qi:qi+Bq] = sim.max(3)[0].sum(1).cpu()
    return out

def timed(fn, n_warm=2, n_run=3):
    for _ in range(n_warm): fn(); sync()
    ts = []
    for _ in range(n_run):
        t0 = time.perf_counter(); fn(); sync()
        ts.append(time.perf_counter() - t0)
    return min(ts)

log("=" * 70)
log("RQ M=16 (deterministic CPU k-means)")
log("=" * 70)

page = torch.load(CACHE / "page_embeddings.pt", weights_only=True).float()
query = torch.load(CACHE / "query_embeddings.pt", weights_only=True).float()
rel = torch.load(CACHE / "relevance.pt", weights_only=True)
P, T, d = page.shape
Q = query.shape[0]
pmask = page.norm(dim=-1) > 1e-6
flat_all = page.reshape(P*T, d)
mask_all = pmask.reshape(P*T)
valid = flat_all[mask_all]
avg_tok = pmask.sum(1).float().mean().item()

log(f"P={P} T={T} d={d} Q={Q} valid={valid.shape[0]:,} avg_tok={avg_tok:.1f}")

# FP16 baseline
fp16_flat = flat_all.clone().to(DEVICE)
fp16_flat[~mask_all.to(DEVICE)] = 0.0
sc_fp16 = maxsim(query, fp16_flat, mask_all, P, T)
ndcg_fp16 = ndcg(sc_fp16, rel)
log(f"FP16 nDCG@10 = {ndcg_fp16:.4f}")

# Fit RQ
log("Fitting RQ M=16...")
t0 = time.time()
cb = fit_rq(valid.to(DEVICE), M_VAL, K_VAL, seed=42)
t_fit = time.time() - t0
log(f"Fit done: {t_fit:.1f}s")

# Encode + reconstruct
log("Encoding...")
t0 = time.time()
codes = encode_rq(valid.to(DEVICE), cb)
t_enc = time.time() - t0

log("Reconstructing...")
t0 = time.time()
rec_valid = reconstruct_rq(codes, cb)
t_dec = time.time() - t0

rq_flat = torch.zeros(P*T, d, device=DEVICE)
rq_flat[mask_all.to(DEVICE)] = rec_valid
rq_flat[~mask_all.to(DEVICE)] = 0.0

sc_rq = maxsim(query, rq_flat, mask_all, P, T)
ndcg_rq = ndcg(sc_rq, rel)
quality_pct = ndcg_rq / ndcg_fp16 * 100

# Latency
box = [None]
def fn(): box[0] = maxsim(query, rq_flat, mask_all, P, T)
t_score = timed(fn)

kb = avg_tok * M_VAL / 1024

result = dict(
    method="rq_m16",
    M=M_VAL, K=K_VAL,
    ndcg_fp16=round(ndcg_fp16, 4),
    ndcg_rq=round(ndcg_rq, 4),
    quality_pct=round(quality_pct, 2),
    storage_kb_per_page=round(kb, 1),
    fit_time_s=round(t_fit, 1),
    encode_time_s=round(t_enc, 3),
    decode_time_s=round(t_dec, 3),
    score_time_s=round(t_score, 3),
)

log(f"\nResult: nDCG@10 = {ndcg_rq:.4f} ({quality_pct:.2f}% of FP16)")
log(f"  Fit={t_fit:.1f}s  Encode={t_enc:.2f}s  Decode={t_dec:.2f}s  Score={t_score:.3f}s")
(OUT / f"rq_m16_{CACHE.name}.json").write_text(json.dumps(result, indent=2))
log(f"Saved: {OUT / 'rq_m16.json'}")
