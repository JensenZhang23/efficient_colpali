#!/usr/bin/env python3
"""
PTM with REAL EOS attention (ColQwen2-v1.0) vs variance proxy — physics.

Why ColQwen2-v1.0: it is the exact model that produced the cached
`experiment/embeddings/colqwen2_v1.0_v3/physics/page_embeddings.pt` (T=731), so the attention
rows align 1:1 with the embedding rows. Its forward() already supports
`output_attentions=True` and returns (proj, self_attns).

Stage 1 (prune):  keep tokens with score >= mean + k*std   (score = attention or variance)
Stage 2 (merge):  per-page k-means over the kept tokens, n_clusters = kept // merge_factor
                  (this is the ORIGINAL prune_merge/prune_then_merge.py semantics)

Usage:
  ASCEND_RT_VISIBLE_DEVICES=9 python3 tree_improvements/methods/ptm/ptm_real_attention.py
  # attention is cached after the first run; later runs reuse it
"""
import os, sys, time, json, io
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
import numpy as np
import pyarrow as pa
from PIL import Image

import argparse as _ap
_p = _ap.ArgumentParser(add_help=True)
_p.add_argument("--cache-dir", type=Path,
    default=ROOT / "experiment/embeddings/colqwen2_v1.0_v3/physics",
    help="Embedding cache dir (page_embeddings.pt, query_embeddings.pt, relevance.pt)")
_p.add_argument("--corpus-arrow", type=Path,
    default=ROOT / ".cache/huggingface/vidore___vidore_v3_physics/corpus/main/dd494effa8ed/data.arrow",
    help="Path to corpus Arrow file (needed for attention extraction)")
_p.add_argument("--model-base", type=Path,
    default=ROOT / ".cache/huggingface/hub/models--vidore--colqwen2-base/snapshots/9fe8a713422a7cb4ef79ca77a09b381ee2243101",
    help="Path to colqwen2-base snapshot dir")
_p.add_argument("--model-adpt", type=Path,
    default=ROOT / ".cache/huggingface/hub/models--vidore--colqwen2-v1.0/snapshots/2b6ac8fb37f46a49e4841e599583d00ae8a20117",
    help="Path to colqwen2-v1.0 adapter snapshot dir")
_p.add_argument("--out", type=Path,
    default=ROOT / "experiment/results/quantization_comparison/ptm_real_attention.json",
    help="Output JSON path")
_p.add_argument("--device", default="npu:0")
_args = _p.parse_args()

DEVICE  = _args.device
CACHE   = _args.cache_dir
CORPUS  = _args.corpus_arrow
BASE    = _args.model_base
ADPT    = _args.model_adpt
ATT_PT  = CACHE / "eos_attention.pt"
OUT     = _args.out
OUT.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# scoring (identical to the other benchmarks)
# --------------------------------------------------------------------------
def sync():
    torch.npu.synchronize()


@torch.no_grad()
def maxsim_score(query, pages, page_mask, bq=8):
    P, T, d = pages.shape
    Q, Tq, _ = query.shape
    neg = torch.finfo(torch.float32).min
    flat = pages.reshape(P * T, d).to(DEVICE)
    mask = page_mask.reshape(P * T).to(DEVICE)
    flat[~mask] = 0.0
    scores = torch.zeros(Q, P)
    for qi in range(0, Q, bq):
        qb = query[qi:qi + bq].to(DEVICE)
        Bq = qb.shape[0]
        sim = qb.reshape(Bq * Tq, d) @ flat.T
        sim = sim.reshape(Bq, Tq, P, T)
        sim.masked_fill_(~mask.view(1, 1, P, T), neg)
        scores[qi:qi + Bq] = sim.max(3)[0].sum(1).cpu()
    return scores


def ndcg(sc, rel, k=10):
    k2 = min(k, sc.shape[1])
    order = sc.argsort(1, descending=True)[:, :k2]
    r = rel.float().gather(1, order)
    disc = 1.0 / torch.log2(torch.arange(1, k2 + 1, dtype=torch.float32) + 1)
    dcg = (r * disc).sum(1)
    idcg = (rel.float().sort(1, descending=True)[0][:, :k2] * disc).sum(1)
    m = idcg > 0
    return (dcg[m] / idcg[m]).mean().item()


def timed_score(query, pages, page_mask, n_warm=1, n_run=3):
    sc = [None]

    def fn():
        sc[0] = maxsim_score(query, pages, page_mask)
        sync()

    for _ in range(n_warm):
        fn()
    ts = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sc[0], min(ts)


# --------------------------------------------------------------------------
# attention extraction
# --------------------------------------------------------------------------
def extract_attention():
    """Run ColQwen2-v1.0 over the physics corpus, keep the EOS attention row.

    Returns dict with two (P, T) float32 tensors:
      'last' — last layer, mean over heads
      'mean' — mean over all layers, mean over heads
    """
    from peft import PeftModel
    from colpali_engine.models.qwen2.colqwen2.modeling_colqwen2 import ColQwen2
    from colpali_engine.models.qwen2.colqwen2.processing_colqwen2 import ColQwen2Processor

    print(f"Loading ColQwen2-v1.0 (base={BASE.name}) ...", flush=True)
    proc = ColQwen2Processor.from_pretrained(str(ADPT))
    model = ColQwen2.from_pretrained(str(BASE), torch_dtype=torch.bfloat16,
                                     attn_implementation="eager")
    model = PeftModel.from_pretrained(model, str(ADPT)).merge_and_unload()
    model = model.to(DEVICE).eval()
    print("model ready", flush=True)

    with pa.ipc.open_stream(str(CORPUS)) as r:
        table = r.read_all()
    rows = table.column("image").to_pylist()
    P = len(rows)
    print(f"corpus: {P} pages", flush=True)

    last_rows, mean_rows, T_ref = [], [], None
    t0 = time.time()
    for i, cell in enumerate(rows):
        img = (Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
               if isinstance(cell, dict) else cell.convert("RGB"))
        batch = {k: (v.to(DEVICE) if hasattr(v, "to") else v)
                 for k, v in proc.process_images([img]).items()}

        with torch.no_grad():
            _, attns = model(**batch, output_attentions=True)
            # each attns[l]: (1, heads, T, T) -> EOS row = [-1]
            eos_last = attns[-1][0, :, -1, :].float().mean(0)          # (T,)
            eos_mean = torch.stack([a[0, :, -1, :].float().mean(0)
                                    for a in attns]).mean(0)           # (T,)

        last_rows.append(eos_last.cpu())
        mean_rows.append(eos_mean.cpu())
        if T_ref is None:
            T_ref = eos_last.numel()

        if (i + 1) % 100 == 0 or i + 1 == P:
            el = time.time() - t0
            print(f"  {i+1}/{P}  ({el:.0f}s, {el/(i+1):.2f}s/page)", flush=True)

    T = max(r.numel() for r in last_rows)
    att_last = torch.zeros(P, T)
    att_mean = torch.zeros(P, T)
    for i, (a, b) in enumerate(zip(last_rows, mean_rows)):
        att_last[i, :a.numel()] = a
        att_mean[i, :b.numel()] = b

    out = {"last": att_last, "mean": att_mean}
    torch.save(out, ATT_PT)
    print(f"saved attention -> {ATT_PT}  shape={tuple(att_last.shape)}", flush=True)
    return out


# --------------------------------------------------------------------------
# PTM
# --------------------------------------------------------------------------
@torch.no_grad()
def _kmeans(x, k, n_iter=10, seed=0):
    """Vectorised euclidean k-means (same semantics as kmeans_simple)."""
    n = x.shape[0]
    if k >= n:
        return x.clone()
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(n, generator=g)[:k]].clone()
    xn = (x * x).sum(1)
    for _ in range(n_iter):
        cn = (c * c).sum(1)
        a = (xn.unsqueeze(1) - 2 * (x @ c.T) + cn).argmin(1)
        nc = torch.zeros_like(c)
        ct = torch.zeros(k)
        nc.index_add_(0, a, x)
        ct.index_add_(0, a, torch.ones(n))
        empty = ct == 0
        nc = nc / ct.clamp_min(1).unsqueeze(1)
        if empty.any():                       # keep old centroid for empty clusters
            nc[empty] = c[empty]
        if (nc - c).norm() < 1e-4:
            c = nc
            break
        c = nc
    return c


def build_ptm(page, scores_PT, prune_k, merge_factor):
    """PTM: prune by `scores_PT`, then per-page k-means merge."""
    P, T, d = page.shape
    kept = []
    for i in range(P):
        pg = page[i]
        vm = pg.norm(dim=-1) > 1e-6
        v = pg[vm].float()
        n = v.shape[0]
        if n == 0:
            kept.append(torch.zeros(1, d))
            continue

        s = scores_PT[i][:T][vm]                       # align scores to valid tokens
        thr = s.mean() + prune_k * s.std()
        keep = s >= thr
        if not keep.any():
            keep[s.argmax()] = True
        pruned = v[keep]

        k = max(1, pruned.shape[0] // merge_factor)
        merged = pruned if k >= pruned.shape[0] else _kmeans(pruned, k)
        kept.append(merged.to(page.dtype))

    L = max(t.shape[0] for t in kept)
    out = torch.zeros(P, L, d, dtype=page.dtype)
    for i, t in enumerate(kept):
        out[i, :t.shape[0]] = t
    return out


def variance_scores(page):
    """Variance proxy used by the original experiments, as a (P, T) tensor."""
    P, T, _ = page.shape
    s = page.float().var(dim=-1)          # (P, T)
    s[page.norm(dim=-1) <= 1e-6] = 0.0
    return s


# --------------------------------------------------------------------------
def main():
    print("=" * 78, flush=True)
    print(f"PTM: real EOS attention vs variance proxy   device={DEVICE}", flush=True)
    print("=" * 78, flush=True)

    page = torch.load(CACHE / "page_embeddings.pt", weights_only=True).float()
    query = torch.load(CACHE / "query_embeddings.pt", weights_only=True).float()
    rel = torch.load(CACHE / "relevance.pt", weights_only=True)
    P, T, d = page.shape
    mask0 = page.norm(dim=-1) > 1e-6
    print(f"pages={P} T={T} d={d} queries={query.shape[0]}", flush=True)

    sc0, t0 = timed_score(query, page, mask0)
    fp16 = ndcg(sc0, rel)
    print(f"FP16 baseline: nDCG@10={fp16:.4f}  score={t0:.3f}s\n", flush=True)

    if ATT_PT.exists():
        print(f"loading cached attention {ATT_PT}", flush=True)
        att = torch.load(ATT_PT, weights_only=True)
    else:
        att = extract_attention()

    score_sets = {
        "variance":  variance_scores(page),
        "attn_last": att["last"],
        "attn_mean": att["mean"],
    }

    configs = [(-1.5, 4), (-1.0, 6), (-0.5, 8), (0.0, 8), (0.5, 8)]
    results = []

    for name, S in score_sets.items():
        print(f"--- PTM ({name}) ---", flush=True)
        for pk, mf in configs:
            tb = time.time()
            pages_c = build_ptm(page, S, pk, mf)
            tb = time.time() - tb
            mk = pages_c.norm(dim=-1) > 1e-6
            sc, ts = timed_score(query, pages_c, mk)
            n = ndcg(sc, rel)
            tok = mk.sum(1).float().mean().item()
            r = dict(score=name, prune_k=pk, merge_factor=mf,
                     ndcg=round(n, 4), quality_pct=round(n / fp16 * 100, 2),
                     avg_tokens=round(tok, 1),
                     storage_kb_page=round(tok * d * 2 / 1024, 2),
                     score_time_s=round(ts, 3), build_time_s=round(tb, 1))
            results.append(r)
            print(f"  k={pk:+.1f} mf={mf:<2}  nDCG={n:.4f} ({r['quality_pct']:5.1f}%)  "
                  f"tok={tok:6.1f}  {r['storage_kb_page']:5.1f}KB/pg  "
                  f"score={ts:.3f}s  build={tb:.0f}s", flush=True)
        print("", flush=True)

    with open(OUT, "w") as f:
        json.dump({"fp16_ndcg": round(fp16, 4), "results": results}, f, indent=2)
    print(f"saved -> {OUT}\n", flush=True)

    # head-to-head
    print("=" * 78, flush=True)
    print(f"{'config':<16}{'variance':>22}{'attn_last':>22}{'attn_mean':>22}", flush=True)
    print("-" * 78, flush=True)
    for pk, mf in configs:
        cells = []
        for name in ("variance", "attn_last", "attn_mean"):
            r = next(x for x in results
                     if x["score"] == name and x["prune_k"] == pk and x["merge_factor"] == mf)
            cells.append(f"{r['ndcg']:.4f} ({r['quality_pct']:4.1f}%,{r['avg_tokens']:5.0f}t)")
        print(f"k={pk:+.1f} mf={mf:<3}" + "".join(f"{c:>22}" for c in cells), flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
