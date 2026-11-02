#!/usr/bin/env python3
"""
Token pruning/merging benchmark — apples-to-apples vs quantization results.

Methods tested:
  - FP16 exact (baseline)
  - local_merge  B={16,32,64,128,256}: corpus-level k-means → per-page NN assignment
  - topk_variance B={16,32,64,128,256}: keep top-B tokens by variance score
  - topk_norm     B={16,32,64,128,256}: keep top-B tokens by L2 norm
  - PTM (prune k + topk merge): several (prune_k, budget) configs

All evaluated with the same MaxSim + nDCG@10 path as the quantization experiments
(all-language queries, 1812 Q × 1674 pages, min-of-3 score timing).

Usage:
  ASCEND_RT_VISIBLE_DEVICES=8 python3 tree_improvements/methods/merge/token_compression_benchmark.py
"""
import os, sys, time, json
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
sys.path.insert(0, str(ROOT))

import torch
import torch_npu
import numpy as np

DEVICE = os.environ.get("DEVICE", "npu:0")
CACHE  = ROOT / "experiment/embeddings/colqwen2_v1.0_v3/physics"
OUT    = ROOT / "experiment/results/quantization_comparison/token_compression_benchmark.json"
OUT.parent.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Scoring — identical to quantization benchmarks
# ---------------------------------------------------------------------------

def sync():
    torch.npu.synchronize()

@torch.no_grad()
def maxsim_score(query, pages, page_mask, bq=8):
    P, T, d = pages.shape
    Q, Tq, _ = query.shape
    neg = torch.finfo(torch.float32).min
    pages_flat = pages.reshape(P*T, d).to(DEVICE)
    mask_flat  = page_mask.reshape(P*T).to(DEVICE)
    pages_flat[~mask_flat] = 0.0
    scores = torch.zeros(Q, P)
    for qi in range(0, Q, bq):
        qb = query[qi:qi+bq].to(DEVICE)
        Bq = qb.shape[0]
        sim = qb.reshape(Bq*Tq, d) @ pages_flat.T
        sim = sim.reshape(Bq, Tq, P, T)
        sim.masked_fill_(~mask_flat.view(1,1,P,T), neg)
        scores[qi:qi+Bq] = sim.max(3)[0].sum(1).cpu()
    return scores

def ndcg(sc, rel, k=10):
    k2 = min(k, sc.shape[1])
    order = sc.argsort(1, descending=True)[:, :k2]
    r    = rel.float().gather(1, order)
    disc = 1.0 / torch.log2(torch.arange(1, k2+1, dtype=torch.float32)+1)
    dcg  = (r * disc).sum(1)
    idcg = (rel.float().sort(1, descending=True)[0][:, :k2] * disc).sum(1)
    m = idcg > 0
    return (dcg[m] / idcg[m]).mean().item()

def timed_score(query, pages, page_mask, n_warm=2, n_run=3):
    sc = [None]
    def fn(): sc[0] = maxsim_score(query, pages, page_mask); sync()
    for _ in range(n_warm): fn()
    ts = []
    for _ in range(n_run):
        t0 = time.perf_counter(); fn()
        ts.append(time.perf_counter() - t0)
    return sc[0], min(ts)

def record(results, fp16_ndcg, method, pages, page_mask, query, rel, t_build):
    sc, t_score = timed_score(query, pages, page_mask)
    n      = ndcg(sc, rel)
    qual   = n / fp16_ndcg * 100
    avg_t  = page_mask.sum(1).float().mean().item()
    kb_pg  = avg_t * pages.shape[2] * 2 / 1024
    entry  = dict(method=method, ndcg=round(n,4), quality_pct=round(qual,2),
                  avg_tokens=round(avg_t,1), storage_kb_page=round(kb_pg,2),
                  score_time_s=round(t_score,3), build_time_s=round(t_build,3),
                  fp16_ndcg=round(fp16_ndcg,4))
    results.append(entry)
    print(f"  {method:<50} nDCG={n:.4f} ({qual:5.1f}%)  "
          f"tok={avg_t:.0f}  {kb_pg:.1f}KB/pg  "
          f"score={t_score:.3f}s  build={t_build:.1f}s", flush=True)
    return entry


# ---------------------------------------------------------------------------
# Fast vectorised k-means (same as rq_quant.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _fast_kmeans(x, k, n_iter=25, seed=0):
    """Vectorised k-means on a (N, d) CPU float tensor."""
    n, d = x.shape
    if k >= n:
        return x.clone()
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(n, generator=g)[:k]].clone()
    xnorm = (x * x).sum(1)
    chunk  = max(1, 2_000_000 // max(1, k))
    for _ in range(n_iter):
        cnorm = (c * c).sum(1)
        a = torch.empty(n, dtype=torch.long)
        for s in range(0, n, chunk):
            xb = x[s:s+chunk]
            d2 = xnorm[s:s+chunk].unsqueeze(1) - 2*(xb @ c.T) + cnorm
            a[s:s+chunk] = d2.argmin(1)
        nc  = torch.zeros_like(c)
        cnt = torch.zeros(k)
        nc.index_add_(0, a, x)
        cnt.index_add_(0, a, torch.ones(n))
        empty = cnt == 0
        nc = nc / cnt.clamp_min(1).unsqueeze(1)
        if empty.any():
            nc[empty] = x[torch.randperm(n, generator=g)[:int(empty.sum())]]
        if (nc - c).norm() < 1e-4:
            break
        c = nc
    return c


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------

def build_local_merge(page, budget):
    """Corpus-level k-means → per-page NN assignment.

    Trains ONE codebook of `budget` centroids on all valid tokens across
    all pages, then for each page selects the `budget` centroids closest to
    that page's token distribution (centroid-to-page NN).  Fast: one k-means
    fit on 1.2M tokens, no per-page loop k-means.
    """
    P, T, d = page.shape
    mask = page.norm(dim=-1) > 1e-6      # (P, T)
    flat  = page.reshape(P*T, d).float()
    valid = flat[mask.reshape(P*T)]      # (N_valid, d)

    print(f"    corpus k-means k={budget} on {valid.shape[0]:,} tokens...", end=" ", flush=True)
    t0 = time.time()
    centroids = _fast_kmeans(valid, budget)   # (budget, d)
    print(f"{time.time()-t0:.1f}s", flush=True)

    # Per-page: assign each valid token to nearest centroid,
    # then keep the `budget` unique centroids that appear in this page
    cnorm  = (centroids * centroids).sum(1)
    out    = torch.zeros(P, budget, d, dtype=page.dtype)
    for i in range(P):
        pg = page[i].float()
        vm = pg.norm(dim=-1) > 1e-6
        v  = pg[vm]
        if v.shape[0] == 0:
            continue
        # assign tokens to centroids
        xn  = (v * v).sum(1)
        d2  = xn.unsqueeze(1) - 2*(v @ centroids.T) + cnorm
        asgn = d2.argmin(1)                        # (N_valid_page,)
        unique_c = asgn.unique()                   # centroids present on this page
        k_page = min(budget, unique_c.shape[0])
        out[i, :k_page] = centroids[unique_c[:k_page]].to(page.dtype)
    return out


def build_topk(page, budget, importance="variance"):
    """Select top-budget tokens per page by variance or norm."""
    P, T, d = page.shape
    out = torch.zeros(P, budget, d, dtype=page.dtype)
    for i in range(P):
        pg   = page[i]
        valid = pg.norm(dim=-1) > 1e-6
        v    = pg[valid]
        n    = v.shape[0]
        if n == 0:
            continue
        k = min(budget, n)
        scores = v.float().var(dim=-1) if importance == "variance" else v.norm(dim=-1)
        idx = scores.topk(k).indices
        out[i, :k] = v[idx]
    return out


def build_tree_kmeans(page, budget, n_iter=10):
    """Per-page k-means tree (uniform k-means, euclidean, same as build_uniform_tree).

    For budget <= 64: use true k-means (centroids). Slow but correct.
    For budget > 64: fall back to topk-norm as a fast approximation.
    """
    P, T, d = page.shape
    out = torch.zeros(P, budget, d, dtype=page.dtype)
    for i in range(P):
        pg    = page[i]
        valid = pg.norm(dim=-1) > 1e-6
        v     = pg[valid].float()
        n     = v.shape[0]
        if n == 0:
            continue
        k = min(budget, n)
        if k == n:
            out[i, :k] = v.to(page.dtype)
        elif budget <= 64:
            # True k-means centroids (per-page)
            c = _fast_kmeans(v, k, n_iter=n_iter)
            out[i, :k] = c.to(page.dtype)
        else:
            # Fast approximation for large budget: topk by norm
            idx = v.norm(dim=-1).topk(k).indices
            out[i, :k] = v[idx].to(page.dtype)
    return out


def build_ptm_kmeans(page, prune_k, budget, n_iter=10):
    """Prune-then-merge with TRUE k-means merge (matching original implementation).

    Stage 1: prune tokens below mean + prune_k * std of variance score.
    Stage 2: k-means clustering on pruned tokens → n_clusters = len(pruned) // budget
             (merge_factor semantics: output ~pruned/budget clusters).

    For merge budget <= 64: true k-means. Else: topk-norm approximation.
    """
    P, T, d = page.shape
    compressed = []
    for i in range(P):
        pg    = page[i]
        valid = pg.norm(dim=-1) > 1e-6
        v     = pg[valid].float()
        n     = v.shape[0]
        if n == 0:
            compressed.append(torch.zeros(1, d))
            continue

        # Stage 1: prune by variance (EOS attention proxy)
        scores = v.var(dim=-1)
        thr    = scores.mean() + prune_k * scores.std()
        keep   = scores >= thr
        if not keep.any():
            keep[scores.argmax()] = True
        pruned = v[keep]

        # Stage 2: k-means merge (output n_clusters ≈ len(pruned) / budget)
        # Here budget is the merge factor — output tokens = pruned / budget
        n_clusters = max(1, pruned.shape[0] // budget)
        k = min(n_clusters, pruned.shape[0])
        if k == pruned.shape[0]:
            merged = pruned
        elif n_clusters <= 64:
            merged = _fast_kmeans(pruned, k, n_iter=n_iter)
        else:
            idx = pruned.norm(dim=-1).topk(k).indices
            merged = pruned[idx]

        compressed.append(merged.to(page.dtype))

    max_len = max(p.shape[0] for p in compressed)
    out = torch.zeros(P, max_len, d, dtype=page.dtype)
    for i, c in enumerate(compressed):
        out[i, :c.shape[0]] = c
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("="*70, flush=True)
    print(f"Token Compression Benchmark  device={DEVICE}", flush=True)
    print("="*70, flush=True)

    page  = torch.load(CACHE/"page_embeddings.pt",  weights_only=True).float()
    query = torch.load(CACHE/"query_embeddings.pt", weights_only=True).float()
    rel   = torch.load(CACHE/"relevance.pt",        weights_only=True)

    P, T, d = page.shape
    Q = query.shape[0]
    mask_orig = page.norm(dim=-1) > 1e-6
    print(f"Pages={P}  T={T}  d={d}  Q={Q}\n", flush=True)

    results = []

    # FP16 baseline
    sc0, t0 = timed_score(query, page, mask_orig)
    fp16_ndcg = ndcg(sc0, rel)
    avg_t0 = mask_orig.sum(1).float().mean().item()
    kb0    = avg_t0 * d * 2 / 1024
    results.append(dict(method="fp16_exact", ndcg=round(fp16_ndcg,4),
                        quality_pct=100.0, avg_tokens=round(avg_t0,1),
                        storage_kb_page=round(kb0,2),
                        score_time_s=round(t0,3), build_time_s=0.0,
                        fp16_ndcg=round(fp16_ndcg,4)))
    print(f"FP16 baseline: nDCG@10={fp16_ndcg:.4f}  avg_tok={avg_t0:.0f}  "
          f"{kb0:.1f} KB/pg  score={t0:.3f}s\n", flush=True)

    print("--- topk_variance ---", flush=True)
    for budget in [256, 128, 64, 32, 16]:
        t0b = time.time()
        pages_c = build_topk(page, budget, "variance")
        t_build = time.time() - t0b
        mask_c  = pages_c.norm(dim=-1) > 1e-6
        record(results, fp16_ndcg, f"topk_variance B={budget}", pages_c, mask_c, query, rel, t_build)

    print("\n--- topk_norm ---", flush=True)
    for budget in [256, 128, 64, 32, 16]:
        t0b = time.time()
        pages_c = build_topk(page, budget, "norm")
        t_build = time.time() - t0b
        mask_c  = pages_c.norm(dim=-1) > 1e-6
        record(results, fp16_ndcg, f"topk_norm B={budget}", pages_c, mask_c, query, rel, t_build)

    print("\n--- PTM (prune + k-means merge, variance proxy attention) ---", flush=True)
    # prune_k: pruning threshold (mean + k*std of variance score)
    # merge_budget: merge_factor — output ~pruned/budget clusters
    ptm_configs = [
        # (prune_k, merge_factor, label_hint)
        (-1.5, 4, "~85tok"),
        (-1.0, 6, "~53tok"),
        (-0.5, 8, "~43tok"),
        ( 0.0, 8, "~30tok"),
        ( 0.5, 8, "~18tok"),
    ]
    for prune_k, merge_factor, hint in ptm_configs:
        t0b = time.time()
        pages_c = build_ptm_kmeans(page, prune_k, merge_factor)
        t_build = time.time() - t0b
        mask_c  = pages_c.norm(dim=-1) > 1e-6
        tag = f"PTM k={prune_k:+.1f} mf={merge_factor} {hint}"
        record(results, fp16_ndcg, tag, pages_c, mask_c, query, rel, t_build)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT}", flush=True)

    # Summary table
    print("\n" + "="*75, flush=True)
    print(f"{'Method':<50} {'nDCG':>6} {'qual%':>6} {'tok':>5} {'KB/pg':>7}", flush=True)
    print("-"*75, flush=True)
    for r in results:
        print(f"  {r['method']:<48} {r['ndcg']:>6.4f} "
              f"{r['quality_pct']:>5.1f}% {r['avg_tokens']:>5.0f} "
              f"{r['storage_kb_page']:>7.1f}", flush=True)
    print("="*75, flush=True)


if __name__ == "__main__":
    main()
