#!/usr/bin/env python3
"""
Extract ViDoRe V3 embeddings with ColQwen2.5-v0.1 (Qwen2.5-VL 3B).

Mirrors tree_improvements/embeddings/colqwen2_v1.0_v3/* (produced with ColQwen2-v1.0),
writing to tree_improvements/embeddings/colqwen2.5_v3/<dataset>/
  page_embeddings.pt   (P, T, 128)  float32, zero-padded
  query_embeddings.pt  (Q, Tq, 128) float32, zero-padded
  relevance.pt         (Q, P)       float32, scores from qrels

Also reports FP16 nDCG@10 per dataset so the numbers are comparable with the
ColQwen2-v1.0 baselines in EXPERIMENT_SUMMARY.md.

Usage:
  ASCEND_RT_VISIBLE_DEVICES=9 python tree_improvements/scripts/encode_colqwen25_v3.py --datasets physics
  ASCEND_RT_VISIBLE_DEVICES=9 python tree_improvements/scripts/encode_colqwen25_v3.py --datasets all
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

def _find_root() -> Path:
    """Walk up from this file to the repo root (the dir holding colpali_engine)."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "colpali_engine").is_dir():
            return parent
    raise RuntimeError(f"repo root not found above {p}")


ROOT = _find_root()
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pyarrow as pa
import torch
import torch_npu  # noqa: F401
from PIL import Image
from peft import PeftModel

from colpali_engine.models.qwen2_5.colqwen2_5.modeling_colqwen2_5 import ColQwen2_5
from colpali_engine.models.qwen2_5.colqwen2_5.processing_colqwen2_5 import ColQwen2_5_Processor

DEVICE = os.environ.get("DEVICE", "npu:0")
MAX_VIS_TOK = int(os.environ.get("MAX_VIS_TOK", "768"))
PAGE_BATCH = int(os.environ.get("PAGE_BATCH", "1"))
QUERY_BATCH = int(os.environ.get("QUERY_BATCH", "32"))

BASE = Path(os.environ.get(
    "MODEL_BASE",
    str(ROOT / ".cache/huggingface/hub/models--vidore--colqwen2.5-base")))
ADPT = Path(os.environ.get(
    "MODEL_ADPT",
    str(ROOT / ".cache/huggingface/hub/models--vidore--colqwen2.5-v0.1/snapshots/4379ecfd4c9446836ad0ef4dd7390b1c748f68d8")))
DS_BASE = Path(os.environ.get(
    "DS_BASE",
    str(ROOT / ".cache/huggingface")))
OUT_BASE = Path(os.environ.get(
    "OUT_BASE",
    str(ROOT / "tree_improvements/embeddings/colqwen2.5_v3")))

ALL_DS = [
    "physics", "computer_science", "energy", "finance_en",
    "finance_fr", "hr", "industrial", "pharmaceuticals",
]


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_arrow_split(ds_root: Path, split: str):
    files = sorted(ds_root.joinpath(split).rglob("*.arrow"))
    if not files:
        raise FileNotFoundError(f"no arrow for {ds_root.name}/{split}")
    tabs = []
    for f in files:
        with pa.ipc.open_stream(str(f)) as r:
            tabs.append(r.read_all())
    return pa.concat_tables(tabs).to_pylist()


def load_dataset(name):
    ds_root = DS_BASE / f"vidore___vidore_v3_{name}"
    corpus = read_arrow_split(ds_root, "corpus")
    queries = read_arrow_split(ds_root, "queries")
    qrels = read_arrow_split(ds_root, "qrels")

    corpus.sort(key=lambda r: r["corpus_id"])
    queries.sort(key=lambda r: r["query_id"])

    c_idx = {r["corpus_id"]: i for i, r in enumerate(corpus)}
    q_idx = {r["query_id"]: i for i, r in enumerate(queries)}
    rel = torch.zeros(len(queries), len(corpus), dtype=torch.float32)
    for row in qrels:
        qi = q_idx.get(row["query_id"])
        ci = c_idx.get(row["corpus_id"])
        if qi is not None and ci is not None:
            rel[qi, ci] = float(row.get("score", 1.0))
    return corpus, queries, rel


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

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
def maxsim(query, page, pmask, bq=8):
    """query (Q,Tq,d) x page (P,T,d) -> scores (Q,P)."""
    Q, Tq, d = query.shape
    P, T = page.shape[0], page.shape[1]
    neg = torch.finfo(torch.float32).min
    pf = page.reshape(P * T, d).to(DEVICE)
    mf = pmask.reshape(P, T).to(DEVICE)
    out = torch.zeros(Q, P)
    for qi in range(0, Q, bq):
        qb = query[qi:qi + bq].to(DEVICE)
        Bq = qb.shape[0]
        sim = qb.reshape(Bq * Tq, d) @ pf.T
        sim = sim.reshape(Bq, Tq, P, T)
        sim.masked_fill_(~mf.view(1, 1, P, T), neg)
        out[qi:qi + Bq] = sim.max(3)[0].sum(1).cpu()
    return out


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_pages(model, proc, corpus):
    embs, mask = [], []
    t0 = time.time()
    n = len(corpus)
    for i in range(0, n, PAGE_BATCH):
        chunk = corpus[i:i + PAGE_BATCH]
        imgs = [
            Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            if isinstance(r["image"], dict) else r["image"].convert("RGB")
            for r in chunk
        ]
        batch = proc.process_images(imgs).to(DEVICE)
        out = model(**batch)
        for j in range(out.shape[0]):
            e = out[j].float().cpu()
            keep = e.norm(dim=-1) > 1e-6
            embs.append(e[keep])
        if (i + PAGE_BATCH) % 100 < PAGE_BATCH or i + PAGE_BATCH >= n:
            el = time.time() - t0
            done = min(i + PAGE_BATCH, n)
            log(f"    pages {done}/{n}  {el:.0f}s ({el/done*1000:.0f} ms/page)")
    T = max(e.shape[0] for e in embs)
    d = embs[0].shape[1]
    page = torch.zeros(n, T, d)
    pmask = torch.zeros(n, T, dtype=torch.bool)
    for i, e in enumerate(embs):
        page[i, :e.shape[0]] = e
        pmask[i, :e.shape[0]] = True
    return page, pmask


@torch.no_grad()
def encode_queries(model, proc, queries):
    embs = []
    t0 = time.time()
    n = len(queries)
    for i in range(0, n, QUERY_BATCH):
        chunk = [r["query"] for r in queries[i:i + QUERY_BATCH]]
        batch = proc.process_queries(chunk).to(DEVICE)
        out = model(**batch)
        for j in range(out.shape[0]):
            e = out[j].float().cpu()
            keep = e.norm(dim=-1) > 1e-6
            embs.append(e[keep])
        if (i + QUERY_BATCH) % 200 < QUERY_BATCH or i + QUERY_BATCH >= n:
            log(f"    queries {min(i+QUERY_BATCH, n)}/{n}  {time.time()-t0:.0f}s")
    Tq = max(e.shape[0] for e in embs)
    d = embs[0].shape[1]
    query = torch.zeros(n, Tq, d)
    for i, e in enumerate(embs):
        query[i, :e.shape[0]] = e
    return query


# ---------------------------------------------------------------------------

def run_dataset(name, model, proc, results):
    log(f"\n{'='*70}\n  {name}\n{'='*70}")
    out_dir = OUT_BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus, queries, rel = load_dataset(name)
    log(f"corpus={len(corpus)} queries={len(queries)} rel_pairs={int((rel>0).sum())}")

    t0 = time.time()
    if (out_dir / "page_embeddings.pt").exists():
        page = torch.load(out_dir / "page_embeddings.pt", map_location="cpu")
        log(f"  page embeddings cached {tuple(page.shape)}")
    else:
        log("  encoding pages ...")
        page, pmask = encode_pages(model, proc, corpus)
        torch.save(page, out_dir / "page_embeddings.pt")
        torch.save(pmask, out_dir / "page_mask.pt")
        log(f"  page {tuple(page.shape)}  ({time.time()-t0:.0f}s)")

    t0 = time.time()
    if (out_dir / "query_embeddings.pt").exists():
        query = torch.load(out_dir / "query_embeddings.pt", map_location="cpu")
        log(f"  query embeddings cached {tuple(query.shape)}")
    else:
        log("  encoding queries ...")
        query = encode_queries(model, proc, queries)
        torch.save(query, out_dir / "query_embeddings.pt")
        log(f"  query {tuple(query.shape)}  ({time.time()-t0:.0f}s)")

    torch.save(rel, out_dir / "relevance.pt")

    pmask = (page.norm(dim=-1) > 1e-6)
    sc = maxsim(query, page, pmask)
    score = ndcg(sc, rel, 10)
    avg_tok = pmask.sum(1).float().mean().item()
    log(f"  *** {name}: ColQwen2.5 FP16 nDCG@10 = {score:.4f}  (avg_tok/page={avg_tok:.0f}) ***")

    results[name] = {
        "ndcg10": round(score, 4),
        "pages": len(corpus),
        "queries": len(queries),
        "avg_tok_per_page": round(avg_tok, 1),
        "page_shape": list(page.shape),
        "query_shape": list(query.shape),
    }
    (OUT_BASE / "summary.json").write_text(json.dumps(results, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Extract ColQwen2.5 embeddings for ViDoRe V3 datasets."
    )
    ap.add_argument(
        "--datasets", default="all",
        help="'all' or comma-separated subset, e.g. physics,energy",
    )
    ap.add_argument(
        "--model-base", type=Path, default=None,
        help="Path to colqwen2.5-base snapshot dir (overrides MODEL_BASE env / compiled default)",
    )
    ap.add_argument(
        "--model-adpt", type=Path, default=None,
        help="Path to colqwen2.5-v0.1 adapter snapshot dir (overrides MODEL_ADPT env / compiled default)",
    )
    ap.add_argument(
        "--data-dir", type=Path, default=None,
        help="Root HF cache dir containing vidore___vidore_v3_* subdirs (overrides DS_BASE env)",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output dir for embeddings (overrides OUT_BASE env)",
    )
    args = ap.parse_args(argv)

    # Allow CLI args to override module-level path constants
    global BASE, ADPT, DS_BASE, OUT_BASE
    if args.model_base is not None:
        BASE = args.model_base
    if args.model_adpt is not None:
        ADPT = args.model_adpt
    if args.data_dir is not None:
        DS_BASE = args.data_dir
    if args.out_dir is not None:
        OUT_BASE = args.out_dir

    names = ALL_DS if args.datasets == "all" else args.datasets.split(",")

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    results = {}
    sf = OUT_BASE / "summary.json"
    if sf.exists():
        results = json.loads(sf.read_text())

    log(f"Loading processor (max_num_visual_tokens={MAX_VIS_TOK}) ...")
    proc = ColQwen2_5_Processor.from_pretrained(str(ADPT), max_num_visual_tokens=MAX_VIS_TOK)

    log(f"Loading ColQwen2.5 base on {DEVICE} ...")
    model = ColQwen2_5.from_pretrained(
        str(BASE), torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    model = PeftModel.from_pretrained(model, str(ADPT)).merge_and_unload()
    model = model.to(DEVICE).eval()
    log("model ready")

    for name in names:
        try:
            run_dataset(name, model, proc, results)
        except Exception as e:
            log(f"  !! {name} FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    log("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
