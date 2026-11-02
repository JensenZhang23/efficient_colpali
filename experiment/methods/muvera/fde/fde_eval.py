#!/usr/bin/env python3
"""
MUVERA FDE (Fixed Dimensional Encoding) on ColPali embeddings — NPU implementation.

What FDE does
-------------
MaxSim needs the full (Q, Tq, P, T) similarity tensor. FDE instead maps each
multi-vector set to ONE fixed-dim vector such that a single dot product
approximates Chamfer/MaxSim similarity:

    S(q, d) ~= <FDE_query(q), FDE_doc(d)>

Space is partitioned by SimHash into 2^k buckets; doc vectors in a bucket are
AVERAGEd, query vectors are SUMmed. Repeating with r independent random
partitionings and concatenating reduces LSH variance. Output dim is
    r * 2^k * proj_dim
optionally count-sketched down to `final_projection_dimension`.

This is a LATENCY method, not a compression method: at the reference config
(r=20, k=7, identity d=128) the FDE is 327,680 floats = 1.28 MB/page, which is
~7x LARGER than the FP16 multi-vector page it replaces. The storage knob is
final_projection_dimension.

Faithfulness
------------
Validated against muvera-py-master/fde_generator.py reference (pure numpy) --
run with VALIDATE=1 to check partition indices and FDE values agree.

Differences from the reference, all pure performance (same math):
  * gray code: vectorized over points (reference loops per point)
  * aggregation: single index_add_ (reference does np.add.at per dim, 128x)
  * empty-partition fill: batched Hamming argmin (reference loops per pair)
  * count sketch: accumulated per repetition, so the full r*2^k*d FDE is never
    materialized

Usage
-----
  CACHE_DIR=$PWD/experiment/embeddings/colqwen2.5_v3/physics \
  ASCEND_RT_VISIBLE_DEVICES=8 \
  python3 experiment/methods/muvera/fde/fde_eval.py

Env
---
  CACHE_DIR   embedding cache dir
  DEVICE      logical NPU (default npu:0)
  VALIDATE    1 = run reference cross-check first (default 1)
  SWEEP       comma-separated config names, or "all" (default all)
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

DEVICE = os.environ.get("DEVICE", "npu:0")
VALIDATE = os.environ.get("VALIDATE", "1") == "1"
SWEEP = os.environ.get("SWEEP", "all")
SEED = int(os.environ.get("SEED", "42"))
CACHE = Path(os.environ.get(
    "CACHE_DIR",
    str(ROOT / "experiment/embeddings/colqwen2.5_v3/physics")))
OUT_DIR = ROOT / "experiment/results/fde"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Random matrices — must match the reference bit-for-bit
# ---------------------------------------------------------------------------

def simhash_matrix(d, k, seed):
    """Reference: rng.normal(0, 1, size=(d, k)) with default_rng(seed)."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(d, k))


def ams_matrix(d, proj_dim, seed):
    """Reference _ams_projection_matrix_from_seed: one +/-1 per input row."""
    rng = np.random.default_rng(seed)
    m = np.zeros((d, proj_dim), dtype=np.float32)
    cols = rng.integers(0, proj_dim, size=d)
    signs = rng.choice([-1.0, 1.0], size=d)
    m[np.arange(d), cols] = signs
    return m


def count_sketch_params(in_dim, out_dim, seed):
    """Reference _apply_count_sketch_to_vector: same seed for every vector,
    so this is one shared linear map. Draw order matters: integers then choice."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, out_dim, size=in_dim)
    signs = rng.choice([-1.0, 1.0], size=in_dim)
    return idx.astype(np.int64), signs.astype(np.float32)


# ---------------------------------------------------------------------------
# Gray-code partition index, vectorized over points
# ---------------------------------------------------------------------------

def gray_partition_index(bits):
    """bits: (N, k) int64 in {0,1}. Returns (N,) int64 partition index.

    Reference per-point loop:
        idx = 0
        for val in sketch: idx = (idx << 1) + ((val > 0) ^ (idx & 1))
    Same recurrence, vectorized across N.
    """
    n, k = bits.shape
    idx = torch.zeros(n, dtype=torch.int64, device=bits.device)
    for j in range(k):
        idx = (idx << 1) + (bits[:, j] ^ (idx & 1))
    return idx


def gray_to_binary_np(num):
    """Reference _gray_code_to_binary."""
    mask = num >> 1
    while mask != 0:
        num = num ^ mask
        mask >>= 1
    return num


def partition_target_bits(num_partitions, k):
    """For empty-partition fill: the bit pattern each partition index decodes to.
    Reference builds binary_array = (gray_to_binary(i) >> arange(k-1,-1,-1)) & 1
    """
    out = np.zeros((num_partitions, k), dtype=np.int64)
    shifts = np.arange(k - 1, -1, -1)
    for i in range(num_partitions):
        out[i] = (gray_to_binary_np(i) >> shifts) & 1
    return out


# ---------------------------------------------------------------------------
# FDE generation (batched, on device)
# ---------------------------------------------------------------------------

def fde_batch(pts, group_id, n_groups, k_sim, n_reps, proj_dim_cfg,
              mode, fill_empty, final_dim, seed, device, T=None,
              chunk_groups=256):
    """
    pts:        (N, d) float32 on device -- valid vectors only, grouped by doc
    group_id:   (N,) int64 -- which doc each vector belongs to
    n_groups:   number of docs
    mode:       "average" (docs) or "sum" (queries)
    proj_dim_cfg: None for identity, else AMS sketch target dim
    final_dim:  None or count-sketch output dim
    T:          max tokens per group, required when fill_empty is True

    Returns (n_groups, out_dim) float32 on device.
    """
    N, d = pts.shape
    num_part = 2 ** k_sim
    proj_dim = d if proj_dim_cfg is None else proj_dim_cfg
    rep_dim = num_part * proj_dim
    full_dim = n_reps * rep_dim

    if final_dim:
        cs_idx_np, cs_sign_np = count_sketch_params(full_dim, final_dim, seed)
        cs_idx = torch.from_numpy(cs_idx_np).to(device)
        cs_sign = torch.from_numpy(cs_sign_np).to(device)
        out = torch.zeros(n_groups, final_dim, dtype=torch.float32, device=device)
    else:
        out = torch.zeros(n_groups, full_dim, dtype=torch.float32, device=device)

    if fill_empty:
        tgt_bits_all = torch.from_numpy(
            partition_target_bits(num_part, k_sim)).to(device)

    for rep in range(n_reps):
        cur_seed = seed + rep

        sh = torch.from_numpy(
            simhash_matrix(d, k_sim, cur_seed).astype(np.float32)).to(device)
        sketches = pts @ sh
        bits = (sketches > 0).to(torch.int64)
        part = gray_partition_index(bits)

        if proj_dim_cfg is None:
            proj = pts
        else:
            am = torch.from_numpy(ams_matrix(d, proj_dim, cur_seed)).to(device)
            proj = pts @ am

        slot = group_id * num_part + part
        sums = torch.zeros(n_groups * num_part, proj_dim,
                           dtype=torch.float32, device=device)
        sums.index_add_(0, slot, proj)

        if mode == "average":
            cnt = torch.zeros(n_groups * num_part, dtype=torch.float32,
                              device=device)
            cnt.index_add_(0, slot, torch.ones(N, dtype=torch.float32,
                                               device=device))
            sums = sums / cnt.clamp_min(1.0).unsqueeze(1)

            if fill_empty:
                empty = (cnt == 0).reshape(n_groups, num_part)
                if empty.any():
                    sums = _fill_empty_partitions(
                        sums, empty, bits, proj, group_id, tgt_bits_all,
                        n_groups, num_part, proj_dim, T, device, chunk_groups)

        rep_fde = sums.reshape(n_groups, rep_dim)

        if final_dim:
            lo, hi = rep * rep_dim, (rep + 1) * rep_dim
            out.index_add_(1, cs_idx[lo:hi], rep_fde * cs_sign[lo:hi])
        else:
            out[:, rep * rep_dim:(rep + 1) * rep_dim] = rep_fde

    return out


def _fill_empty_partitions(sums, empty, bits, proj, group_id, tgt_bits_all,
                           n_groups, num_part, proj_dim, T, device,
                           chunk_groups):
    """For each (doc, empty partition), copy the vector of that doc's token whose
    SimHash bits are closest in Hamming distance to the partition's code.

    Reference does one (doc, partition) pair at a time. We batch over docs:
    build (chunk, T, num_part) Hamming distances, mask padding to a large
    value, argmin over T. Same result, no Python loop over pairs.
    """
    BIG = 10 ** 6
    k = bits.shape[1]
    sums3 = sums.reshape(n_groups, num_part, proj_dim)

    pos = torch.arange(group_id.shape[0], device=device)
    first = torch.zeros(n_groups, dtype=torch.int64, device=device)
    first.scatter_reduce_(0, group_id, pos, reduce="amin", include_self=False)
    local = pos - first[group_id]

    bits_d = torch.zeros(n_groups, T, k, dtype=torch.int64, device=device)
    bits_d[group_id, local] = bits
    valid = torch.zeros(n_groups, T, dtype=torch.bool, device=device)
    valid[group_id, local] = True
    proj_d = torch.zeros(n_groups, T, proj_dim, dtype=torch.float32,
                         device=device)
    proj_d[group_id, local] = proj

    for s in range(0, n_groups, chunk_groups):
        e = min(s + chunk_groups, n_groups)
        em = empty[s:e]
        if not em.any():
            continue
        dist = (bits_d[s:e].unsqueeze(2) !=
                tgt_bits_all.view(1, 1, num_part, k)).sum(-1).to(torch.int32)
        dist = dist.masked_fill(~valid[s:e].unsqueeze(2), BIG)
        best = dist.argmin(1)
        picked = torch.gather(
            proj_d[s:e], 1, best.unsqueeze(-1).expand(-1, -1, proj_dim))
        blk = sums3[s:e].clone()
        blk[em] = picked[em]
        sums3[s:e] = blk

    return sums3.reshape(n_groups * num_part, proj_dim)


# ---------------------------------------------------------------------------
# Metrics
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
def maxsim(query, page_flat, pmask, P, T, bq=8):
    """Reference FP16 MaxSim, same harness as the other benchmarks."""
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


def timed(fn, n_warm=1, n_run=3):
    for _ in range(n_warm):
        fn(); torch.npu.synchronize()
    ts = []
    for _ in range(n_run):
        t0 = time.perf_counter(); fn(); torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)


# ---------------------------------------------------------------------------
# Reference cross-check
# ---------------------------------------------------------------------------

def validate_against_reference(device):
    """Compare our batched device FDE against the pure-numpy reference."""
    from fde_generator import (
        FixedDimensionalEncodingConfig, ProjectionType,
        generate_query_fde, generate_document_fde,
    )
    log("\n" + "-" * 70)
    log("Validating against muvera-py-master/fde_generator.py reference")
    log("-" * 70)

    rng = np.random.default_rng(0)
    d, k_sim, n_reps = 128, 4, 3
    cases = [
        ("query  identity", "sum", False, None, None),
        ("doc    identity", "average", True, None, None),
        ("doc    AMS16", "average", True, 16, None),
        ("doc    final1024", "average", True, None, 1024),
    ]
    all_ok = True
    for name, mode, fill, ams, final in cases:
        n_tok = 37
        pts_np = rng.normal(size=(n_tok, d)).astype(np.float32)

        cfg = FixedDimensionalEncodingConfig(
            dimension=d, num_repetitions=n_reps, num_simhash_projections=k_sim,
            seed=SEED, fill_empty_partitions=fill,
            final_projection_dimension=final,
        )
        if ams:
            cfg = FixedDimensionalEncodingConfig(
                dimension=d, num_repetitions=n_reps,
                num_simhash_projections=k_sim, seed=SEED,
                fill_empty_partitions=fill,
                projection_type=ProjectionType.AMS_SKETCH,
                projection_dimension=ams,
                final_projection_dimension=final,
            )
        ref = (generate_document_fde(pts_np, cfg) if mode == "average"
               else generate_query_fde(pts_np, cfg))

        pts = torch.from_numpy(pts_np).to(device)
        gid = torch.zeros(n_tok, dtype=torch.int64, device=device)
        ours = fde_batch(pts, gid, 1, k_sim, n_reps, ams, mode, fill,
                         final, SEED, device, T=n_tok)[0].cpu().numpy()

        if ours.shape != ref.shape:
            log(f"  {name:<18} SHAPE MISMATCH ours={ours.shape} ref={ref.shape}")
            all_ok = False
            continue
        denom = max(np.abs(ref).max(), 1e-8)
        rel_err = np.abs(ours - ref).max() / denom
        ok = rel_err < 2e-4
        all_ok &= ok
        log(f"  {name:<18} dim={ours.shape[0]:<7} max_rel_err={rel_err:.2e}  "
            f"{'OK' if ok else 'MISMATCH'}")
    log(f"\n  Reference agreement: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------------
# Sweep configs
# ---------------------------------------------------------------------------
# name -> (n_reps, k_sim, ams_proj_dim, final_dim)
# storage per page = out_dim * 4 B (float32), matching the reference dtype.
CONFIGS = {
    # reference config from muvera-py-master/main.py
    "r20_k7_id":        (20, 7, None, None),
    "r20_k6_id":        (20, 6, None, None),
    "r10_k6_id":        (10, 6, None, None),
    "r5_k6_id":         (5,  6, None, None),
    # inner AMS projection shrinks proj_dim 128 -> 16
    "r20_k6_ams16":     (20, 6, 16,   None),
    "r10_k6_ams16":     (10, 6, 16,   None),
    # final count-sketch projection: the real storage knob
    "r20_k6_fp8192":    (20, 6, None, 8192),
    "r20_k6_fp4096":    (20, 6, None, 4096),
    "r20_k6_fp2048":    (20, 6, None, 2048),
    "r20_k6_fp1024":    (20, 6, None, 1024),
}


def main():
    log("=" * 78)
    log(f"MUVERA FDE on ColPali   cache={CACHE.name}   device={DEVICE}")
    log("=" * 78)

    device = DEVICE

    if VALIDATE:
        if not validate_against_reference(device):
            log("\n!! reference validation failed -- numbers below are suspect")

    # ---- load embeddings -------------------------------------------------
    page = torch.load(CACHE / "page_embeddings.pt", weights_only=True).float()
    query = torch.load(CACHE / "query_embeddings.pt", weights_only=True).float()
    rel = torch.load(CACHE / "relevance.pt", weights_only=True)
    P, T, d = page.shape
    Q, Tq, _ = query.shape
    pmask = page.norm(dim=-1) > 1e-6
    qmask = query.norm(dim=-1) > 1e-6
    log(f"\nP={P} T={T} d={d}  Q={Q} Tq={Tq}  "
        f"avg_tok/page={pmask.sum(1).float().mean():.1f}")

    # ---- FP16 MaxSim baseline -------------------------------------------
    pf = page.reshape(P * T, d).clone().to(device)
    pf[~pmask.reshape(P * T).to(device)] = 0.0
    sc_ref = [None]

    def run_maxsim():
        sc_ref[0] = maxsim(query, pf, pmask, P, T)

    t_maxsim = timed(run_maxsim)
    ndcg_fp16 = ndcg(sc_ref[0], rel)
    kb_fp16 = pmask.sum(1).float().mean().item() * d * 2 / 1024
    log(f"\nFP16 MaxSim   nDCG@10={ndcg_fp16:.4f}   score={t_maxsim:.3f}s   "
        f"{kb_fp16:.1f} KB/page")
    del pf
    torch.npu.empty_cache()

    # ---- flatten to grouped point clouds ---------------------------------
    pmask_f = pmask.reshape(P * T)
    pts_doc = page.reshape(P * T, d)[pmask_f].to(device)
    gid_doc = torch.arange(P).unsqueeze(1).expand(P, T).reshape(-1)[pmask_f]
    gid_doc = gid_doc.to(device)

    qmask_f = qmask.reshape(Q * Tq)
    pts_q = query.reshape(Q * Tq, d)[qmask_f].to(device)
    gid_q = torch.arange(Q).unsqueeze(1).expand(Q, Tq).reshape(-1)[qmask_f]
    gid_q = gid_q.to(device)

    names = list(CONFIGS) if SWEEP == "all" else SWEEP.split(",")
    results = []

    for name in names:
        n_reps, k_sim, ams, final = CONFIGS[name]
        out_dim = final or (n_reps * (2 ** k_sim) * (ams or d))
        log("\n" + "-" * 78)
        log(f"[{name}]  reps={n_reps} k_sim={k_sim} (2^{k_sim}={2**k_sim} part) "
            f"ams={ams} final={final}  ->  dim={out_dim:,}")

        try:
            t0 = time.time()
            fde_d = fde_batch(pts_doc, gid_doc, P, k_sim, n_reps, ams,
                              "average", True, final, SEED, device, T=T)
            torch.npu.synchronize()
            t_build_d = time.time() - t0

            t0 = time.time()
            fde_q = fde_batch(pts_q, gid_q, Q, k_sim, n_reps, ams,
                              "sum", False, final, SEED, device, T=Tq)
            torch.npu.synchronize()
            t_build_q = time.time() - t0

            sc_box = [None]

            def run_fde():
                # .cpu() included so this matches the MaxSim baseline, which
                # accumulates its (Q, P) result on the host. At single-digit ms
                # the 12 MB D2H is a real fraction of the total.
                sc_box[0] = (fde_q @ fde_d.T).cpu()

            t_score = timed(run_fde)
            nd = ndcg(sc_box[0], rel)

            kb_page = out_dim * 4 / 1024
            row = dict(
                config=name, n_reps=n_reps, k_sim=k_sim, num_partitions=2 ** k_sim,
                ams_proj=ams, final_dim=final, out_dim=out_dim,
                ndcg=round(nd, 4),
                quality_pct=round(nd / ndcg_fp16 * 100, 2),
                kb_per_page=round(kb_page, 1),
                storage_vs_fp16=round(kb_page / kb_fp16, 3),
                score_s=round(t_score, 4),
                speedup_vs_maxsim=round(t_maxsim / t_score, 2),
                build_doc_s=round(t_build_d, 1),
                build_query_s=round(t_build_q, 1),
            )
            results.append(row)
            log(f"  nDCG@10={nd:.4f} ({row['quality_pct']:.2f}% of FP16)   "
                f"score={t_score:.4f}s ({row['speedup_vs_maxsim']:.2f}x)   "
                f"{kb_page:.1f} KB/page ({row['storage_vs_fp16']:.2f}x FP16)")
            log(f"  build: doc {t_build_d:.1f}s  query {t_build_q:.1f}s")

            del fde_d, fde_q, sc_box
            torch.npu.empty_cache()
        except RuntimeError as e:
            log(f"  !! FAILED: {type(e).__name__}: {str(e)[:200]}")
            torch.npu.empty_cache()

    # ---- summary ---------------------------------------------------------
    log("\n" + "=" * 78)
    log(f"SUMMARY   {CACHE.name}   FP16 MaxSim = {ndcg_fp16:.4f} @ "
        f"{t_maxsim:.3f}s, {kb_fp16:.1f} KB/page")
    log("=" * 78)
    log(f"{'config':<17}{'dim':>9}{'nDCG':>8}{'vs FP16':>9}"
        f"{'KB/pg':>9}{'score s':>9}{'speedup':>9}")
    log("-" * 78)
    for r in results:
        log(f"{r['config']:<17}{r['out_dim']:>9,}{r['ndcg']:>8.4f}"
            f"{r['quality_pct']:>8.2f}%{r['kb_per_page']:>9.1f}"
            f"{r['score_s']:>9.4f}{r['speedup_vs_maxsim']:>8.2f}x")
    log("=" * 78)

    out = OUT_DIR / f"fde_sweep_{CACHE.name}.json"
    out.write_text(json.dumps(dict(
        cache=CACHE.name, seed=SEED,
        P=P, T=T, d=d, Q=Q, Tq=Tq,
        ndcg_fp16=round(ndcg_fp16, 4),
        maxsim_score_s=round(t_maxsim, 3),
        fp16_kb_per_page=round(kb_fp16, 1),
        configs=results,
    ), indent=2))
    log(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
