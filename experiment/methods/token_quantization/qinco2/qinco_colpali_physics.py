#!/usr/bin/env python3
"""
QINCo V2 on ColPali Physics Embeddings — single-NPU, 10-epoch experiment.

Trains QINCo V2 neural quantizer on already-extracted ColQwen2 page embeddings
(no ColQwen2 re-inference). Evaluates retrieval quality (nDCG@10), compression
ratio, and encode/decode latency.

Usage:
  ASCEND_RT_VISIBLE_DEVICES=8 python tree_improvements/methods/qinco2/qinco_colpali_physics.py
"""
import os
import sys
import time
import json
import tempfile
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
_p_ext.add_argument("--qinco-dir", type=Path, default=ROOT / "experiment/methods/token_quantization/qinco2",
    help="Path to Qinco-main directory (must contain qinco/ package)")
_ext_args, _ = _p_ext.parse_known_args()
sys.path.insert(0, str(getattr(_ext_args, "qinco_dir")))

import numpy as np
import torch
import torch_npu

DEVICE = os.environ.get("DEVICE", "npu:0")   # logical device after ASCEND_RT_VISIBLE_DEVICES remapping
EPOCHS = int(os.environ.get("EPOCHS", "10"))
M_VAL  = int(os.environ.get("M_VAL", "16"))   # number of codebooks
CACHE = Path(os.environ.get("CACHE_DIR", str(ROOT / "experiment/embeddings/colqwen2_v1.0_v3/physics")))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(ROOT / "experiment/results/qinco_colqwen2_v1.0")))
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Scoring helpers (same as existing benchmarks)
# ---------------------------------------------------------------------------

def sync():
    torch.npu.synchronize()


@torch.no_grad()
def score_fp16(query, page_flat, mask_flat, P, T, bq=8):
    Q, Tq, d = query.shape
    neg = torch.finfo(torch.float32).min
    scores = torch.zeros(Q, P)
    for qi in range(0, Q, bq):
        qb = query[qi:qi+bq].to(DEVICE)
        Bq = qb.shape[0]
        sim = qb.reshape(Bq*Tq, d) @ page_flat.T
        sim = sim.reshape(Bq, Tq, P, T)
        sim.masked_fill_(~mask_flat.view(1, 1, P, T), neg)
        scores[qi:qi+Bq] = sim.max(3)[0].sum(1).cpu()
    return scores


@torch.no_grad()
def score_reconstructed(query, rec_flat, mask_flat, P, T, bq=8):
    return score_fp16(query, rec_flat, mask_flat, P, T, bq)


def ndcg(sc, rel, k=10):
    k2 = min(k, sc.shape[1])
    order = sc.argsort(1, descending=True)[:, :k2]
    r = rel.float().gather(1, order)
    disc = 1.0 / torch.log2(torch.arange(1, k2+1, dtype=torch.float32) + 1)
    dcg = (r * disc).sum(1)
    idcg = (rel.float().sort(1, descending=True)[0][:, :k2] * disc).sum(1)
    m = idcg > 0
    return (dcg[m] / idcg[m]).mean().item()


def timed(fn, n_warm=2, n_run=3):
    for _ in range(n_warm):
        fn(); sync()
    ts = []
    for _ in range(n_run):
        t0 = time.perf_counter(); fn(); sync()
        ts.append(time.perf_counter() - t0)
    return min(ts)


# ---------------------------------------------------------------------------
# Build QINCo config + model the way QINCo expects
# ---------------------------------------------------------------------------

def build_qinco_cfg_and_model(d, M, K, epochs):
    """
    Construct a properly-initialised (cfg, model) pair for a single-device
    QINCo V2 training run on ColPali embeddings.

    Key insight: SharedCfgState stores _-prefixed attrs on the Python wrapper
    object, not in the OmegaConf dict, so they must be set AFTER wrapping.
    """
    from omegaconf import OmegaConf
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from qinco.utils import SharedCfgState
    from qinco.model import QINCo, initialize_qinco_codebooks
    from qinco.vrq import TorchRQ

    raw = OmegaConf.create({
        # architecture
        "M": M, "K": K, "L": 2, "dh": 256, "de": 64,
        "A": 16, "B": 4,
        # training
        "optimizer": "adamw", "lr": 8e-4, "wd": 0.1,
        "batch": 1024, "epochs": epochs,
        "grad_clip": 0.1, "grad_accumulate": 1,
        # data
        "ds": {"trainset": None, "valset": 10_000, "loop": 10_000_000},
        # misc
        "seed": 0, "verbose": True,
        "codebook_noise_init": 0.1, "enc_max_bs": 65536,
        "mse_scale": 1, "qinco1_mode": False,
        "ivf_in_use": False, "task": "train",
        "cpu": False, "inference": False,
        "output": None, "model": None,
        "tensorboard": None, "ivf_centroids": None,
        "index": None, "encoded_db": None,
        "encoded_trainset": None, "pairwise_decoder": None,
        "db": None, "trainset": None, "queries": None, "queries_gt": None,
        "encode_trainset": False,
        "default_datasets": {},
        "env": {},
        "scheduler": {
            "name": "cosine", "ramp_epochs": 3,
            "lr_min_fact": 1e-3, "stop_patience": 10,
        },
    })

    # wrap BEFORE setting _-prefixed runtime attrs
    cfg = SharedCfgState(raw)

    # accelerator (single-device, fp16)
    from torch_npu.contrib import transfer_to_npu
    accelerator = Accelerator(mixed_precision="fp16", cpu=False)
    cfg._accelerator = accelerator

    # runtime config attrs required by QINCo internals
    cfg._D = d
    cfg._M_ivf = M
    cfg._ivf_book = None
    cfg._ivf_centroids_preloaded = None
    cfg._optimizer = None
    cfg._scheduler = None
    cfg._cur_epoch = 0
    cfg._qinco_jit = False   # avoid torch.jit issues on NPU

    set_seed(cfg.seed)
    return cfg, accelerator


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print(f"QINCo V2 on ColPali Physics  M={M_VAL}  epochs={EPOCHS}  device={DEVICE}")
    print("=" * 70)

    # ---- load embeddings -------------------------------------------------
    print("Loading embeddings...", flush=True)
    page  = torch.load(CACHE / "page_embeddings.pt",  weights_only=True).float()
    query = torch.load(CACHE / "query_embeddings.pt", weights_only=True).float()
    rel   = torch.load(CACHE / "relevance.pt",        weights_only=True)

    P, T, d = page.shape
    Q = query.shape[0]
    page_mask = page.norm(dim=-1) > 1e-6
    flat_all  = page.reshape(P*T, d)
    mask_all  = page_mask.reshape(P*T)
    valid     = flat_all[mask_all]           # (N_valid, d)
    N_valid   = valid.shape[0]
    print(f"Pages={P}  T={T}  d={d}  Q={Q}  valid_tokens={N_valid:,}", flush=True)

    # ---- train / val split (numpy, for QINCo) ----------------------------
    perm    = torch.randperm(N_valid)
    n_train = int(N_valid * 0.9)
    train_np = valid[perm[:n_train]].numpy()
    val_np   = valid[perm[n_train:]].numpy()
    print(f"Train={len(train_np):,}  Val={len(val_np):,}", flush=True)

    # ---- build config + model  -------------------------------------------
    M, K = M_VAL, 256
    cfg, accelerator = build_qinco_cfg_and_model(d, M, K, EPOCHS)
    device = accelerator.device

    # data_mean / data_std needed for model init
    cfg._data_mean = train_np.mean(axis=0)                  # (d,)
    cfg._data_std  = float(train_np.std())

    print("Initialising codebooks with static RQ...", flush=True)
    from qinco.vrq import apply_rq_on_vectors
    xt_t   = torch.from_numpy(train_np)
    xval_t = torch.from_numpy(val_np)
    _, _, rq_centroids = apply_rq_on_vectors(cfg, xt_t, xval_t)

    from qinco.model import QINCo, initialize_qinco_codebooks
    model = QINCo(cfg).to(device)
    # rq_centroids are CPU tensors; move them to device before init
    rq_centroids_dev = [c.to(device) for c in rq_centroids]
    initialize_qinco_codebooks(cfg, model, rq_centroids_dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"QINCo model  params={n_params/1e6:.2f}M", flush=True)

    # ---- dataset + dataloader  -------------------------------------------
    class NdArr(torch.utils.data.Dataset):
        def __init__(self, arr): self.t = torch.from_numpy(arr).float()
        def __len__(self): return len(self.t)
        def __getitem__(self, i): return self.t[i]

    train_loader = torch.utils.data.DataLoader(
        NdArr(train_np), batch_size=cfg.batch, shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        NdArr(val_np), batch_size=cfg.batch, shuffle=False,
        num_workers=0, drop_last=False,
    )

    # ---- optimizer / scheduler  ------------------------------------------
    from qinco.qinco_tasks import build_optimizer, build_scheduler
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # ---- training loop  --------------------------------------------------
    print("\n" + "-"*70, flush=True)
    for epoch in range(EPOCHS):
        cfg._cur_epoch = epoch
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            # QINCo's "train" step returns (codes, xhat, loss_dict)
            codes, xhat, losses = model(batch, step="train")
            loss = sum(losses.values()) if isinstance(losses, dict) else losses

            optimizer.zero_grad()
            accelerator.backward(loss)
            if cfg.grad_clip:
                accelerator.clip_grad_value_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
            if n_batches * cfg.batch >= cfg.ds.loop:
                break

        # validation MSE
        model.eval()
        val_mse, val_n = 0.0, 0
        with torch.no_grad():
            for vb in val_loader:
                vb = vb.to(device)
                codes_v, xhat_v, _ = model(vb, step="train")
                val_mse += ((vb - xhat_v) ** 2).sum().item()
                val_n   += vb.numel()
        val_mse /= val_n

        scheduler.step(epoch)
        dt = time.time() - t0
        print(f"Epoch {epoch+1:2d}/{EPOCHS}  "
              f"train_loss={total_loss/n_batches:.5f}  "
              f"val_mse={val_mse:.5f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"time={dt:.0f}s", flush=True)

    # ---- encode all page tokens ------------------------------------------
    print("\nEncoding all page tokens...", flush=True)
    model.eval()
    flat_npu  = flat_all.to(device)
    mask_npu  = mask_all.to(device)
    flat_npu[~mask_npu] = 0.0

    enc_bs    = 4096
    all_codes = []   # list of M-element code-lists per chunk
    t_enc0    = time.time()
    with torch.no_grad():
        for i in range(0, P*T, enc_bs):
            batch  = flat_npu[i:i+enc_bs]
            codes  = model(batch, step="encode")   # list of M tensors (B,)
            all_codes.append([c.cpu() for c in codes])
    t_enc = time.time() - t_enc0

    # reassemble to list[M] of (P*T,) tensors
    page_codes = [torch.cat([chunk[m] for chunk in all_codes]) for m in range(M)]
    print(f"Encode done: {t_enc:.2f}s  ({t_enc/(P*T)*1e6:.2f} µs/token)", flush=True)

    # ---- decode + score --------------------------------------------------
    print("Decoding & scoring...", flush=True)
    rec_flat = torch.zeros(P*T, d, device=device)
    dec_bs   = 4096
    t_dec0   = time.time()
    with torch.no_grad():
        for i in range(0, P*T, dec_bs):
            chunk_codes = [page_codes[m][i:i+dec_bs].to(device) for m in range(M)]
            rec_flat[i:i+dec_bs] = model(chunk_codes, step="decode")
    t_dec = time.time() - t_dec0
    rec_flat[~mask_npu] = 0.0

    sc_fp16  = score_fp16(query, flat_npu, mask_npu, P, T)
    sc_qinco = score_reconstructed(query, rec_flat, mask_npu, P, T)
    ndcg_fp16  = ndcg(sc_fp16,  rel)
    ndcg_qinco = ndcg(sc_qinco, rel)
    quality_pct = ndcg_qinco / ndcg_fp16 * 100

    # test MSE
    test_tokens = valid[perm[n_train:]].to(device)
    with torch.no_grad():
        tc = model(test_tokens, step="encode")
        tr = model(tc, step="decode")
        test_mse = ((test_tokens - tr) ** 2).mean().item()

    storage_b_tok = M   # K=256 → 1 byte/codebook

    # ---- latency (full-index score, min of 3) ----------------------------
    sc_ref = [None]
    def fn_fp16():  sc_ref[0] = score_fp16(query, flat_npu, mask_npu, P, T)
    def fn_qinco(): sc_ref[0] = score_reconstructed(query, rec_flat, mask_npu, P, T)
    t_fp16  = timed(fn_fp16)
    t_qinco = timed(fn_qinco)

    # ---- print + save  ---------------------------------------------------
    result = dict(
        method="qinco_v2_M16_10ep",
        M=M, K=K, L=2, dh=256, de=64, epochs=EPOCHS,
        params_M=round(n_params/1e6, 3),
        ndcg_fp16=round(ndcg_fp16,  4),
        ndcg_qinco=round(ndcg_qinco, 4),
        quality_pct=round(quality_pct, 2),
        test_mse=round(test_mse, 6),
        storage_b_tok=storage_b_tok,
        compression_ratio=round(256.0 / storage_b_tok, 1),
        encode_time_s=round(t_enc, 3),
        decode_time_s=round(t_dec, 3),
        encode_us_per_token=round(t_enc / (P*T) * 1e6, 2),
        score_fp16_s=round(t_fp16,  3),
        score_qinco_s=round(t_qinco, 3),
        latency_ratio=round(t_qinco / t_fp16, 3),
    )

    print("\n" + "=" * 70)
    print("Results:")
    for k, v in result.items():
        print(f"  {k:<30} {v}")
    print("=" * 70)

    out_json = OUT_DIR / "qinco_colpali_physics_M16_10ep.json"
    out_model = OUT_DIR / "qinco_colpali_physics_M16_10ep.pt"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"state_dict": accelerator.unwrap_model(model).state_dict(),
                "result": result}, out_model)
    print(f"\nSaved:  {out_json}")
    print(f"        {out_model}")


if __name__ == "__main__":
    main()
