#!/usr/bin/env python3
"""RQ-MoE vs static RQ on ViDoRe physics: does an input-dependent (learned MoE)
residual quantizer beat static k-means RQ at the SAME 16 B/tok, measured by
nDCG@10 (not just MSE)?

RQ-MoE (ICML'26, KDEGroup) refines each residual codeword through a hierarchical
MoE conditioned on an accumulated "instruction" stream, so the codebook adapts
per input. We reuse its model as-is (RQ-MoE-main/model_rqmoe.py) but replace its
FAISS centroid init with our own fit_rq and drop CUDA/DDP so it runs on NPU.

Storage is identical to static RQ (M codes/token); only the decode is heavier
(gate + N experts per step). This script answers quality first; latency later.

Usage:
  DATASET=physics DEVICE=npu:0 EPOCHS=1 python3 tree_improvements/methods/rq_moe/rqmoe_physics.py
Env: M K N L H BATCH LR EPOCHS VAL_FRAC
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

import argparse as _ap_ext
_p_ext = _ap_ext.ArgumentParser(add_help=False)
_p_ext.add_argument("--rqmoe-dir", type=Path, default=ROOT / "experiment/methods/token_quantization/rq_moe",
    help="Path to RQ-MoE-main directory (contains model_rqmoe.py)")
_ext_args, _ = _p_ext.parse_known_args()
sys.path.insert(0, str(getattr(_ext_args, "rqmoe_dir")))

import numpy as np
import torch
import torch_npu  # noqa: F401
from torch import optim

from experiments.methods.token_quantization.rq.rq_quant import fit_rq, encode_rq, reconstruct_rq
import model_rqmoe

DEVICE = os.environ.get("DEVICE", "npu:0")
DATASET = os.environ.get("DATASET", "physics")
M = int(os.environ.get("M", "16"))
K = int(os.environ.get("K", "256"))
N = int(os.environ.get("N", "4"))
L = int(os.environ.get("L", "2"))
H = int(os.environ.get("H", "256"))
BATCH = int(os.environ.get("BATCH", "4096"))
LR = float(os.environ.get("LR", "1e-3"))
EPOCHS = int(os.environ.get("EPOCHS", "150"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.1"))
PATIENCE = int(os.environ.get("PATIENCE", "40"))
EVALBS = int(os.environ.get("EVALBS", "32768"))

TAG = os.environ.get("TAG", "")
CACHE = ROOT / "experiment/embeddings/colqwen2_v1.0_v3" / DATASET
OUT = ROOT / "experiment/results/quantization_comparison" / f"rqmoe_{DATASET}{('_' + TAG) if TAG else ''}.json"
OUT.parent.mkdir(exist_ok=True)


def sync():
    torch.npu.synchronize()


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
def reconstruct_all(model, valid_norm, db_scale, bs=None):
    """Encode+decode every valid token through RQ-MoE, back to original scale."""
    if bs is None:
        bs = EVALBS
    out = torch.empty_like(valid_norm)
    for i in range(0, valid_norm.shape[0], bs):
        codes, xhat = model.encode(valid_norm[i:i + bs])
        out[i:i + bs] = xhat
    return out * db_scale


def main():
    print("=" * 70, flush=True)
    print(f"RQ-MoE vs static RQ  dataset={DATASET}  M={M} K={K} N={N} L={L} H={H}device={DEVICE}", flush=True)
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
    Nvalid = valid.shape[0]
    print(f"Pages={P} T={T} d={d} Q={Q}  valid tokens={Nvalid:,}", flush=True)

    # ── preload scoring tensors ─────────────────────────────────────────────
    pf_npu = flat.to(DEVICE); pf_npu[~mask.to(DEVICE)] = 0.0
    mf_npu = mask.to(DEVICE)

    # ── FP16 upper bound ─────────────────────────────────────────────────────
    sc = score_fp16(query, pf_npu, mf_npu, P, T)
    ndcg_fp16 = ndcg(sc, rel)
    print(f"FP16 exact              nDCG@10={ndcg_fp16:.4f}", flush=True)

    # ── static RQ baseline (original scale) ───────────────────────────────────
    print("Fitting static RQ...", flush=True)
    t0 = time.time()
    valid_npu = valid.to(DEVICE)
    cb = fit_rq(valid_npu, num_codebooks=M, codebook_size=K, n_iter=25, seed=0)
    print(f"  fit {time.time() - t0:.1f}s", flush=True)
    codes_v = encode_rq(valid_npu, cb)
    rec_v = reconstruct_rq(codes_v, cb)
    rec_flat = torch.zeros(P * T, d, device=DEVICE)
    rec_flat[mf_npu] = rec_v
    sc = score_fp16(query, rec_flat, mf_npu, P, T)
    ndcg_static = ndcg(sc, rel)
    mse_static = ((rec_v - valid_npu) ** 2).sum(1).mean().item()
    print(f"static RQ M={M}       nDCG@10={ndcg_static:.4f} ({ndcg_static/ndcg_fp16*100:.2f}%)  MSE={mse_static:.4e}", flush=True)
    del rec_flat, rec_v, codes_v

    # ── build RQ-MoE, init from static RQ centroids ────────────────────────────
    db_scale = float(valid.abs().max())
    rq_centroids = np.stack([c.cpu().numpy() for c in cb])  # (M, K, d)
    model = model_rqmoe.RQMoE(d=d, K=K, M=M, N=N, L=L, H=H, dropout=0.1).to(DEVICE)
    model.db_scale = db_scale
    with torch.no_grad():
        model.codebook0.weight.copy_(torch.from_numpy(rq_centroids[0]).to(DEVICE) / db_scale)
        for i, step in enumerate(model.steps):
            step.codebook.weight.copy_(torch.from_numpy(rq_centroids[i + 1]).to(DEVICE) / db_scale)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RQ-MoE params: {nparam:,}", flush=True)

    # ── train/val split (normalized space) ─────────────────────────────────────
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(Nvalid, generator=g)
    nval = int(Nvalid * VAL_FRAC)
    val_idx = perm[:nval]
    tr_idx = perm[nval:]
    valid_norm = (valid / db_scale)
    xval = valid_norm[val_idx].to(DEVICE)
    xt = valid_norm[tr_idx].to(DEVICE)
    print(f"train={xt.shape[0]:,}  val={xval.shape[0]:,}", flush=True)

    @torch.no_grad()
    def val_mse():
        model.eval()
        s, n = 0.0, 0
        for i in range(0, xval.shape[0], EVALBS):
            b = xval[i:i + EVALBS]
            _, xhat = model.encode(b)
            s += ((xhat - b) ** 2).sum().item()
            n += b.shape[0]
        model.train()
        return s / n * db_scale ** 2

    best = float("inf"); best_state = None; since = 0; lr = LR
    print(f"\n--- training (max {EPOCHS} epochs, patience {PATIENCE}) ---", flush=True)
    v0 = val_mse()
    print(f"epoch   0  val MSE={v0:.4e}  (init = static RQ)", flush=True)
    for epoch in range(1, EPOCHS + 1):
        te = time.time()
        rs = np.random.RandomState(epoch)
        bperm = torch.from_numpy(rs.permutation(xt.shape[0])).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        for i in range(0, xt.shape[0], BATCH):
            b = xt[bperm[i:i + BATCH]]
            model.zero_grad()
            codes, xhat, losses = model(b)
            eps = 1e-6
            orig_e = (b ** 2).sum() + eps
            nl = [torch.log1p(losses[0] / orig_e)]
            for layer in range(1, len(losses)):
                nl.append(torch.log1p(losses[layer] / (losses[layer - 1].detach() + eps)))
            loss = torch.stack(nl).sum()
            loss.backward()
            opt.step()
        sync()
        v = val_mse()
        dt = time.time() - te
        flag = ""
        if v < best:
            best = v; since = 0
            best_state = {k2: v2.detach().cpu().clone() for k2, v2 in model.state_dict().items()}
            flag = " *"
        else:
            since += 1
            if since >= PATIENCE:
                lr /= 10
                since = 0
                flag = f"  lr->{lr:g}"
                if lr < 5e-6:
                    print(f"epoch {epoch:3d}  val MSE={v:.4e}  {dt:.1f}s  lr too small, stop", flush=True)
                    break
        print(f"epoch {epoch:3d}  val MSE={v:.4e}  {dt:.1f}s{flag}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── final nDCG@10 with RQ-MoE reconstruction ───────────────────────────────
    model.eval()
    valid_norm_npu = valid_norm.to(DEVICE)
    rec_v = reconstruct_all(model, valid_norm_npu, db_scale)
    mse_moe = ((rec_v - valid_npu) ** 2).sum(1).mean().item()
    rec_flat = torch.zeros(P * T, d, device=DEVICE)
    rec_flat[mf_npu] = rec_v
    sc = score_fp16(query, rec_flat, mf_npu, P, T)
    ndcg_moe = ndcg(sc, rel)
    print(f"\n{'='*70}", flush=True)
    print(f"FP16 exact     nDCG@10={ndcg_fp16:.4f}  (100.00%)", flush=True)
    print(f"static RQ M={M}   nDCG@10={ndcg_static:.4f}  ({ndcg_static/ndcg_fp16*100:.2f}%)  MSE={mse_static:.4e}", flush=True)
    print(f"RQ-MoE  M={M}     nDCG@10={ndcg_moe:.4f}  ({ndcg_moe/ndcg_fp16*100:.2f}%)  MSE={mse_moe:.4e}", flush=True)
    print(f"delta nDCG (MoE - static): {ndcg_moe - ndcg_static:+.4f}", flush=True)

    payload = dict(dataset=DATASET, device=DEVICE, M=M, K=K, N=N, L=L, H=H,
                   epochs_run=epoch, best_val_mse=best,
                   ndcg_fp16=round(ndcg_fp16, 4),
                   ndcg_static=round(ndcg_static, 4), mse_static=mse_static,
                   ndcg_moe=round(ndcg_moe, 4), mse_moe=mse_moe,
                   delta_ndcg=round(ndcg_moe - ndcg_static, 4),
                   timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n✓ saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
