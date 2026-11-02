#!/usr/bin/env python3
"""
Two-stage retrieval: FDE recall -> exact/RQ MaxSim rerank.

Why this can beat single-stage on BOTH axes
-------------------------------------------
Latency: MaxSim's cost is Q*Tq*P*T. Stage 1 (a single dense matmul) cuts P
from the whole corpus to K candidates, so stage 2 pays Q*Tq*K*T. With K=100
out of 1674 pages that is a 16x reduction in the expensive part.

Quality: MaxSim scores a page as sum_i max_j (q_i . d_j). A page holding one
token that is broadly similar to every query token gets rewarded by every
term in the sum -- a known false-positive mode. FDE averages within SimHash
partitions, so it does not reward that spike. If stage 1 drops such pages
before stage 2 ever sees them, cascade nDCG can EXCEED full FP16 MaxSim.
This is the same effect measured in diag_score_decomposition.py, where
quantization-induced argmax flips were net positive (gap_flip = -0.0066).

Storage is additive and reported as such
----------------------------------------
The FDE index is an EXTRA cost on top of whatever stage 2 stores:
    total = FDE_KB/page + rerank_KB/page
So the interesting stage-1 configs are the compact ones (count-sketched down
to 4-8k dims), not the 1280 KB/page reference config, which alone is 7x
larger than the FP16 page it replaces.

Usage
-----
  CACHE_DIR=$PWD/experiment/embeddings/colqwen2.5_v3/physics \
  ASCEND_RT_VISIBLE_DEVICES=8 \
  python3 experiment/methods/muvera/fde/fde_two_stage.py

Env
---
  CACHE_DIR    embedding cache dir
  DEVICE       logical NPU (default npu:0)
  STAGE1       comma-separated FDE config names (default a compact subset)
  KS           comma-separated candidate counts (default 10,25,50,100,200,400)
  RERANK       exact,rq  (default both)
  FUSION       1 = also test score fusion (default 1)
"""
import os
import sys
import json
import time
from pathlib import Path


def _find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "experiment" / "__init__.py").is_file():
            return parent
    raise RuntimeError(f"experiment package root not found above {p}")


ROOT = _find_root()
sys.path.insert(0, str(ROOT))

import argparse as _ap_ext
_p_ext = _ap_ext.ArgumentParser(add_help=False)
_p_ext.add_argument("--muvera-dir", type=Path, default=ROOT / "experiment/methods/muvera",
    help="Path to muvera-py-master directory (contains fde_generator.py)")
_ext_args, _ = _p_ext.parse_known_args()
sys.path.insert(0, str(getattr(_ext_args, "muvera_dir")))

import numpy as np
import torch
import torch_npu  # noqa: F401

from experiment.methods.muvera.fde.fde_eval import (
    fde_batch, ndcg, maxsim, timed, CONFIGS,
)
from experiment.methods.token_quantization.rq.rq_quant import (
    fit_rq, encode_rq, reconstruct_rq,
)

DEVICE = os.environ.get("DEVICE", "npu:0")
SEED = int(os.environ.get("SEED", "42"))
STAGE1 = os.environ.get(
    "STAGE1", "r20_k7_id,r10_k6_id,r20_k6_ams16,r20_k6_fp8192,r20_k6_fp4096")
KS = [int(x) for x in os.environ.get("KS", "10,25,50,100,200,400").split(",")]
RERANK = os.environ.get("RERANK", "exact,rq").split(",")
FUSION = os.environ.get("FUSION", "1") == "1"
M_VAL = int(os.environ.get("M_VAL", "16"))
K_VAL = int(os.environ.get("K_VAL", "256"))
CACHE = Path(os.environ.get(
    "CACHE_DIR",
    str(ROOT / "experiment/embeddings/colqwen2.5_v3/physics")))
OUT_DIR = ROOT / "experiment/results/fde"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEG = torch.finfo(torch.float32).min


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def rerank_candidates(query, page_2d, pmask_2d, cand, P, T, d, bq=4):
    """Exact MaxSim restricted to per-query candidate sets.

    query:     (Q, Tq, d) cpu
    page_2d:   (P, T*d) on device, padding already zeroed -- page-major layout
    pmask_2d:  (P, T) bool on device
    cand:      (Q, K) long on device -- page ids to score for each query
    Returns (Q, P) cpu with NEG everywhere outside the candidate set, so the
    shared ndcg() sees an unchanged score-matrix shape.

    Gather granularity matters a lot. A page's T tokens are contiguous, so
    indexing (P, T*d) by page id reads K contiguous T*d blocks; indexing
    (P*T, d) by token id would issue K*T tiny 128-float lookups instead.
    Measured on this NPU: 466 GB/s page-major vs 61 GB/s token-major, a 7.6x
    difference that decides whether the cascade is faster than full MaxSim.
    """
    Q, Tq, _ = query.shape
    K = cand.shape[1]
    out = torch.full((Q, P), NEG, dtype=torch.float32)

    for qi in range(0, Q, bq):
        qb = query[qi:qi + bq].to(cand.device)
        B = qb.shape[0]
        cb = cand[qi:qi + B]                                  # (B, K)
        flat = cb.reshape(-1)                                 # (B*K,)

        pg = page_2d[flat].reshape(B, K * T, d)
        mk = pmask_2d[flat].reshape(B, K, T)

        sim = torch.bmm(qb, pg.transpose(1, 2)).reshape(B, Tq, K, T)
        sim.masked_fill_(~mk.unsqueeze(1), NEG)
        sc = sim.max(3)[0].sum(1)                             # (B, K)

        out[qi:qi + B] = out[qi:qi + B].scatter(1, cb.cpu(), sc.cpu())
    return out


@torch.no_grad()
def recall_at_k(cand, rel):
    """Fraction of each query's relevant pages that survive stage 1."""
    Q = rel.shape[0]
    hit, tot = 0.0, 0.0
    cand_cpu = cand.cpu()
    for q in range(Q):
        r = (rel[q] > 0).nonzero(as_tuple=True)[0]
        if r.numel() == 0:
            continue
        keep = torch.isin(r, cand_cpu[q]).sum().item()
        hit += keep
        tot += r.numel()
    return hit / max(tot, 1.0)


def fuse_on_candidates(sc_rerank, fde_scores, cand, alphas):
    """Blend stage-1 and stage-2 scores, restricted to the candidate set.

    Costs nothing at query time: both score vectors are already in hand once
    the cascade has run, so this is pure post-processing on (Q, K) slices.
    Normalisation is over the candidate set only -- the non-candidates are
    NEG and must not enter the mean/std.

    Returns {alpha: (Q, P) fused score matrix}.
    """
    Q = sc_rerank.shape[0]
    cand_cpu = cand.cpu()
    ms = sc_rerank.gather(1, cand_cpu)                 # (Q, K)
    fd = fde_scores.cpu().gather(1, cand_cpu)          # (Q, K)
    zm = (ms - ms.mean(1, keepdim=True)) / ms.std(1, keepdim=True).clamp_min(1e-6)
    zf = (fd - fd.mean(1, keepdim=True)) / fd.std(1, keepdim=True).clamp_min(1e-6)
    out = {}
    for a in alphas:
        fused = torch.full_like(sc_rerank, NEG)
        out[a] = fused.scatter(1, cand_cpu, a * zm + (1 - a) * zf)
    return out


def zscore_rows(x):
    """Per-query normalisation so FDE and MaxSim scores can be added."""
    m = x.mean(1, keepdim=True)
    s = x.std(1, keepdim=True).clamp_min(1e-6)
    return (x - m) / s


def main():
    log("=" * 96)
    log(f"Two-stage FDE recall -> MaxSim rerank   cache={CACHE.name}   device={DEVICE}")
    log("=" * 96)

    device = DEVICE
    page = torch.load(CACHE / "page_embeddings.pt", weights_only=True).float()
    query = torch.load(CACHE / "query_embeddings.pt", weights_only=True).float()
    rel = torch.load(CACHE / "relevance.pt", weights_only=True)
    P, T, d = page.shape
    Q, Tq, _ = query.shape
    pmask = page.norm(dim=-1) > 1e-6
    qmask = query.norm(dim=-1) > 1e-6
    avg_tok = pmask.sum(1).float().mean().item()
    log(f"P={P} T={T} d={d}  Q={Q} Tq={Tq}  avg_tok/page={avg_tok:.1f}")

    pmask_f = pmask.reshape(P * T)
    page_flat = page.reshape(P * T, d).clone().to(device)
    pmask_dev = pmask_f.to(device)
    page_flat[~pmask_dev] = 0.0
    page_2d = page_flat.reshape(P, T * d)      # page-major view for the gather
    pmask_2d = pmask_dev.reshape(P, T)

    # ---- stage-0 baseline: full FP16 MaxSim -------------------------------
    box = [None]

    def run_full():
        box[0] = maxsim(query, page_flat, pmask, P, T)

    t_full = timed(run_full)
    sc_fp16 = box[0]
    ndcg_fp16 = ndcg(sc_fp16, rel)
    kb_fp16 = avg_tok * d * 2 / 1024
    log(f"\n[baseline] full FP16 MaxSim   nDCG@10={ndcg_fp16:.4f}   "
        f"score={t_full:.3f}s   {kb_fp16:.1f} KB/page")

    # ---- RQ M=16 reconstruction for the cheap rerank variant --------------
    rq_flat = None
    kb_rq = 0.0
    if "rq" in RERANK:
        log(f"\nFitting RQ M={M_VAL} for the rerank index ...")
        valid = page[pmask]
        t0 = time.time()
        cb = fit_rq(valid.to(device), M_VAL, K_VAL)
        codes = encode_rq(valid.to(device), cb)
        rec = reconstruct_rq(codes, cb).cpu()
        t_rq_fit = time.time() - t0
        rq_flat = torch.zeros(P * T, d)
        rq_flat[pmask_f] = rec
        rq_flat = rq_flat.to(device)
        rq_flat[~pmask_dev] = 0.0
        rq_2d = rq_flat.reshape(P, T * d)
        kb_rq = avg_tok * M_VAL / 1024
        sc_rq_full = maxsim(query, rq_flat, pmask, P, T)
        log(f"  fit {t_rq_fit:.1f}s   full-RQ MaxSim nDCG={ndcg(sc_rq_full, rel):.4f}"
            f"   {kb_rq:.1f} KB/page")

    # ---- point clouds for FDE --------------------------------------------
    pts_doc = page.reshape(P * T, d)[pmask_f].to(device)
    gid_doc = torch.arange(P).unsqueeze(1).expand(P, T).reshape(-1)[pmask_f].to(device)
    qmask_f = qmask.reshape(Q * Tq)
    pts_q = query.reshape(Q * Tq, d)[qmask_f].to(device)
    gid_q = torch.arange(Q).unsqueeze(1).expand(Q, Tq).reshape(-1)[qmask_f].to(device)

    rows = []
    fusion_rows = []
    cascade_fusion_rows = []

    for name in STAGE1.split(","):
        n_reps, k_sim, ams, final = CONFIGS[name]
        out_dim = final or (n_reps * (2 ** k_sim) * (ams or d))
        kb_fde = out_dim * 4 / 1024
        log("\n" + "=" * 96)
        log(f"STAGE 1 = {name}   dim={out_dim:,}   {kb_fde:.1f} KB/page")
        log("=" * 96)

        fde_d = fde_batch(pts_doc, gid_doc, P, k_sim, n_reps, ams,
                          "average", True, final, SEED, device, T=T)
        fde_q = fde_batch(pts_q, gid_q, Q, k_sim, n_reps, ams,
                          "sum", False, final, SEED, device, T=Tq)

        s1 = [None]

        def run_s1():
            s1[0] = fde_q @ fde_d.T

        t_s1 = timed(run_s1)
        fde_scores = s1[0]
        nd_s1 = ndcg(fde_scores.cpu(), rel)
        log(f"  stage-1 alone: nDCG@10={nd_s1:.4f} ({nd_s1/ndcg_fp16*100:.2f}%)  "
            f"scan={t_s1*1000:.1f}ms")

        # score fusion (no cascade): alpha*maxsim + (1-alpha)*fde
        if FUSION:
            zf = zscore_rows(fde_scores.cpu())
            zm = zscore_rows(sc_fp16)
            best = (None, -1.0)
            for a in [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]:
                nd_f = ndcg(a * zm + (1 - a) * zf, rel)
                if nd_f > best[1]:
                    best = (a, nd_f)
                fusion_rows.append(dict(stage1=name, alpha=a, ndcg=round(nd_f, 4),
                                        delta=round(nd_f - ndcg_fp16, 4)))
            log(f"  fusion best: alpha={best[0]}  nDCG={best[1]:.4f}  "
                f"({best[1]-ndcg_fp16:+.4f} vs FP16)")

        for K in KS:
            if K > P:
                continue
            cand = fde_scores.topk(K, dim=1).indices          # (Q, K)
            rec_k = recall_at_k(cand, rel)

            for mode in RERANK:
                src = page_2d if mode == "exact" else rq_2d
                bq = 8 if K <= 100 else (4 if K <= 200 else 2)
                bx = [None]

                def run_2s():
                    c = (fde_q @ fde_d.T).topk(K, dim=1).indices
                    bx[0] = rerank_candidates(query, src, pmask_2d, c,
                                              P, T, d, bq=bq)

                t_2s = timed(run_2s, n_warm=1, n_run=3)
                nd_2s = ndcg(bx[0], rel)
                kb_total = kb_fde + (kb_fp16 if mode == "exact" else kb_rq)
                rows.append(dict(
                    stage1=name, fde_dim=out_dim, K=K, rerank=mode,
                    ndcg=round(nd_2s, 4),
                    quality_pct=round(nd_2s / ndcg_fp16 * 100, 2),
                    delta_vs_fp16=round(nd_2s - ndcg_fp16, 4),
                    recall_at_k=round(rec_k, 4),
                    latency_s=round(t_2s, 4),
                    speedup=round(t_full / t_2s, 2),
                    kb_fde=round(kb_fde, 1),
                    kb_rerank=round(kb_fp16 if mode == "exact" else kb_rq, 1),
                    kb_total=round(kb_total, 1),
                    storage_vs_fp16=round(kb_total / kb_fp16, 3),
                ))
                if FUSION and mode == "exact":
                    alphas = [0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
                    fus = fuse_on_candidates(bx[0], fde_scores, cand, alphas)
                    best_a, best_nd = None, -1.0
                    for a, mtx in fus.items():
                        nd_c = ndcg(mtx, rel)
                        cascade_fusion_rows.append(dict(
                            stage1=name, K=K, alpha=a, ndcg=round(nd_c, 4),
                            delta_vs_fp16=round(nd_c - ndcg_fp16, 4),
                            latency_s=round(t_2s, 4),
                            speedup=round(t_full / t_2s, 2)))
                        if nd_c > best_nd:
                            best_a, best_nd = a, nd_c
                    log(f"  K={K:<4} cascade+fusion  best alpha={best_a}  "
                        f"nDCG={best_nd:.4f} ({best_nd/ndcg_fp16*100:6.2f}%, "
                        f"{best_nd-ndcg_fp16:+.4f})  {t_2s*1000:7.1f}ms "
                        f"({t_full/t_2s:.2f}x)")
                log(f"  K={K:<4} rerank={mode:<5} nDCG={nd_2s:.4f} "
                    f"({nd_2s/ndcg_fp16*100:6.2f}%, {nd_2s-ndcg_fp16:+.4f})  "
                    f"recall@K={rec_k:.3f}  {t_2s*1000:7.1f}ms "
                    f"({t_full/t_2s:6.2f}x)  {kb_total:6.1f} KB/pg")

        del fde_d, fde_q
        torch.npu.empty_cache()

    # ---- summary ----------------------------------------------------------
    log("\n" + "=" * 96)
    log(f"SUMMARY  {CACHE.name}   FP16 MaxSim: nDCG={ndcg_fp16:.4f}, "
        f"{t_full:.3f}s, {kb_fp16:.1f} KB/page")
    log("=" * 96)
    gains = [r for r in rows if r["delta_vs_fp16"] > 0]
    if gains:
        log("\nConfigs that BEAT full FP16 MaxSim:")
        for r in sorted(gains, key=lambda x: -x["delta_vs_fp16"])[:12]:
            log(f"  {r['stage1']:<16} K={r['K']:<4} {r['rerank']:<5} "
                f"nDCG={r['ndcg']:.4f} ({r['delta_vs_fp16']:+.4f})  "
                f"{r['latency_s']*1000:7.1f}ms ({r['speedup']:.2f}x)  "
                f"{r['kb_total']:.1f} KB/pg")
    else:
        log("\nNo config exceeded full FP16 MaxSim.")

    cf_win = [r for r in cascade_fusion_rows if r["delta_vs_fp16"] > 0]
    if cf_win:
        log("\nCASCADE + FUSION beating full FP16 (faster AND better):")
        for r in sorted(cf_win, key=lambda x: -x["delta_vs_fp16"])[:12]:
            log(f"  {r['stage1']:<16} K={r['K']:<4} alpha={r['alpha']:<5} "
                f"nDCG={r['ndcg']:.4f} ({r['delta_vs_fp16']:+.4f})  "
                f"{r['latency_s']*1000:7.1f}ms ({r['speedup']:.2f}x)")

    fast = [r for r in rows if r["speedup"] > 1.0 and r["quality_pct"] >= 99.0]
    if fast:
        log("\nFaster than FP16 AND >=99% quality:")
        for r in sorted(fast, key=lambda x: -x["speedup"])[:12]:
            log(f"  {r['stage1']:<16} K={r['K']:<4} {r['rerank']:<5} "
                f"nDCG={r['ndcg']:.4f} ({r['quality_pct']:.2f}%)  "
                f"{r['latency_s']*1000:7.1f}ms ({r['speedup']:.2f}x)  "
                f"{r['kb_total']:.1f} KB/pg ({r['storage_vs_fp16']:.2f}x)")
    log("=" * 96)

    out = OUT_DIR / f"fde_two_stage_{CACHE.name}.json"
    out.write_text(json.dumps(dict(
        cache=CACHE.name, P=P, T=T, d=d, Q=Q, Tq=Tq, seed=SEED,
        ndcg_fp16=round(ndcg_fp16, 4),
        maxsim_full_s=round(t_full, 3),
        fp16_kb_per_page=round(kb_fp16, 1),
        rq_kb_per_page=round(kb_rq, 1),
        two_stage=rows, fusion=fusion_rows,
        cascade_fusion=cascade_fusion_rows,
    ), indent=2))
    log(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
