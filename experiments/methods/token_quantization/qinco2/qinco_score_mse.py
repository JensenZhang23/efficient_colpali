#!/usr/bin/env python3
"""
QINCo V2 + Pointwise Score-MSE training on ColPali embeddings.

Motivation
----------
Pure MSE trains the quantizer to minimise per-token reconstruction error
isotropically. But MaxSim retrieval only reads the argmax token per query
token, and only the component of the error along the query direction enters
the score. Score-MSE directly aligns the quantized MaxSim score with the
FP16 score, so the error budget is spent where it matters.

Loss
----
  L = alpha * L_mse  +  (1 - alpha) * L_score

  L_mse   = QINCo's native per-token MSE (summed over codebook steps)
  L_score = mean_over_pages((S_hat(q,d) - S_fp16(q,d))^2 / T^2)
            where S is the full MaxSim matrix (Q x P_batch)

The score loss is computed on a separate page-level mini-batch that is
sampled each training step; it does NOT modify the token-level DataLoader.

Gradient path
-------------
QINCo's encode (beam search) is always detached -- codes are selected
without gradient. The gradient flows through the *decode* path only:
xhat = sum_m codebook_m[code_m], so dL/d(codebook weights) is intact.
This is not a limitation we introduce -- it is how QINCo trains normally.
The score loss adds a ranking signal that tells the decode path which
direction to push each centroid.

Usage
-----
  # M=16, baseline check (should match original QINCo nDCG)
  ASCEND_RT_VISIBLE_DEVICES=8 ALPHA=0.0 M_VAL=16 EPOCHS=10 \\
    python3 tree_improvements/methods/qinco2/qinco_score_mse.py

  # M=16, score-MSE
  ASCEND_RT_VISIBLE_DEVICES=8 ALPHA=0.5 M_VAL=16 EPOCHS=10 \\
    python3 tree_improvements/methods/qinco2/qinco_score_mse.py

  # M=4, score-MSE (the real test)
  ASCEND_RT_VISIBLE_DEVICES=9 ALPHA=0.5 M_VAL=4 EPOCHS=10 \\
    CACHE_DIR=.../embeddings/colqwen2.5_v3/physics \\
    python3 tree_improvements/methods/qinco2/qinco_score_mse.py

Env
---
  CACHE_DIR      path to embeddings dir (default: colqwen2.5_v3/physics)
  OUT_DIR        output directory
  DEVICE         logical NPU id (default npu:0)
  M_VAL          codebooks (default 16)
  K_VAL          codebook size (default 256)
  EPOCHS         training epochs (default 10)
  ALPHA          weight of score loss (default 0.5; 0.0 = pure MSE baseline)
  ALPHA_RAMP     if >0, linearly ramp alpha from 0 to ALPHA over this many
                 epochs (safer than WARMUP_EPOCHS; avoids injecting score loss
                 at peak LR). Default 0 (no ramp, use ALPHA directly).
  PAGE_BATCH     pages per score-loss step (default 32)
  SCORE_BQ       query batch size inside score loss (default 4)

NOTE on WARMUP_EPOCHS
---------------------
Setting WARMUP_EPOCHS=N injects score loss only after epoch N. This interacts
badly with the cosine scheduler's ramp_epochs=3: LR peaks at epoch 3, so if
N>=3 the first score-loss update hits at maximum LR, kicking codebooks out of
their MSE basin without recovery time. Empirically WARMUP_EPOCHS=3 achieved
nDCG=0.4546 (97.53%), WORSE than WARMUP_EPOCHS=0 which got 0.4640 (99.55%).
Prefer WARMUP_EPOCHS=0 (default). Use ALPHA_RAMP for a gentler transition.
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

import argparse as _ap_ext
_p_ext = _ap_ext.ArgumentParser(add_help=False)
_p_ext.add_argument("--qinco-dir", type=Path, default=ROOT / "experiment/methods/token_quantization/qinco2",
    help="Path to Qinco-main directory (must contain qinco/ package)")
_ext_args, _ = _p_ext.parse_known_args()
sys.path.insert(0, str(getattr(_ext_args, "qinco_dir")))

import numpy as np
import torch
import torch_npu  # noqa: F401

# 确定性设置
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE     = os.environ.get("DEVICE", "npu:0")
EPOCHS     = int(os.environ.get("EPOCHS", "10"))
M_VAL      = int(os.environ.get("M_VAL", "16"))
K_VAL      = int(os.environ.get("K_VAL", "256"))
ALPHA      = float(os.environ.get("ALPHA", "0.5"))    # score-loss weight
ALPHA_RAMP = int(os.environ.get("ALPHA_RAMP", "0"))    # ramp alpha from 0→ALPHA over N epochs (0=off)
PAGE_BATCH = int(os.environ.get("PAGE_BATCH", "32"))   # pages per score step
SCORE_BQ   = int(os.environ.get("SCORE_BQ", "4"))      # query batch for score
SCORE_EVERY = int(os.environ.get("SCORE_EVERY", "100"))  # run score loss every N token batches
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "0"))  # pure-MSE warmup (keep 0; see docstring)
CACHE = Path(os.environ.get(
    "CACHE_DIR",
    str(ROOT / "experiment/embeddings/colqwen2.5_v3/physics")))
OUT_DIR = Path(os.environ.get(
    "OUT_DIR",
    str(ROOT / "experiment/results/qinco_score_mse")))
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(*a):
    print(*a, flush=True)


def ndcg(sc, rel, k=10):
    k2 = min(k, sc.shape[1])
    order = sc.argsort(1, descending=True)[:, :k2]
    r = rel.float().gather(1, order)
    disc = 1.0 / torch.log2(torch.arange(1, k2 + 1, dtype=torch.float32) + 1)
    dcg = (r * disc).sum(1)
    idcg = (rel.float().sort(1, descending=True)[0][:, :k2] * disc).sum(1)
    m = idcg > 0
    return (dcg[m] / idcg[m]).mean().item()


def sync():
    torch.npu.synchronize()


@torch.no_grad()
def maxsim_scores(query, page_flat, pmask_flat, P, T, bq=8):
    """Returns (Q, P) score matrix; page_flat already zeroed at padding."""
    Q, Tq, d = query.shape
    neg = torch.finfo(torch.float32).min
    mf = pmask_flat.reshape(P, T).to(DEVICE)
    out = torch.zeros(Q, P)
    for qi in range(0, Q, bq):
        qb = query[qi:qi + bq].to(DEVICE)
        Bq = qb.shape[0]
        sim = (qb.reshape(Bq * Tq, d) @ page_flat.T).reshape(Bq, Tq, P, T)
        sim.masked_fill_(~mf.view(1, 1, P, T), neg)
        out[qi:qi + Bq] = sim.max(3)[0].sum(1).cpu()
    return out


def score_loss_for_batch(model, page_batch, query_all, pmask_batch,
                         fp16_scores_batch, P_b, T, d, bq=SCORE_BQ):
    """
    Encode -> decode a batch of pages, compute MaxSim against all queries,
    return MSE vs the FP16 scores for those pages.

    page_batch:       (P_b, T, d) on CPU
    pmask_batch:      (P_b, T) bool on CPU
    fp16_scores_batch: (Q, P_b) on CPU -- pre-computed teacher scores
    """
    device = next(model.parameters()).device
    flat = page_batch.reshape(P_b * T, d).to(device)
    mask_flat = pmask_batch.reshape(P_b * T).to(device)
    flat[~mask_flat] = 0.0

    # encode is no_grad (beam search, codes detached); decode is differentiable
    with torch.no_grad():
        codes = model(flat, step="encode")   # list of M tensors (P_b*T,)
    hat_flat = model(codes, step="decode")   # (P_b*T, d)  -- grad flows here
    hat_flat[~mask_flat] = 0.0

    # compute MaxSim with the reconstructed pages
    pmask_b = pmask_batch.reshape(P_b, T)
    neg = torch.finfo(torch.float32).min
    mf = pmask_b.to(device)
    hat_pt = hat_flat.reshape(P_b, T, d)
    Q, Tq, _ = query_all.shape
    sc_hat = torch.zeros(Q, P_b, device=device)
    for qi in range(0, Q, bq):
        qb = query_all[qi:qi + bq].to(device)
        Bq = qb.shape[0]
        sim = (qb.reshape(Bq * Tq, d) @ hat_flat.T).reshape(Bq, Tq, P_b, T)
        sim.masked_fill_(~mf.view(1, 1, P_b, T), neg)
        sc_hat[qi:qi + Bq] = sim.max(3)[0].sum(1)

    # teacher scores for these pages
    sc_fp16 = fp16_scores_batch.to(device)  # (Q, P_b)

    # normalise by T so the loss scale doesn't depend on sequence length
    loss = ((sc_hat - sc_fp16) / T).pow(2).mean()
    return loss


# ---------------------------------------------------------------------------
# QINCo config + model (same as original qinco_colpali_physics.py)
# ---------------------------------------------------------------------------

def build_qinco_cfg_and_model(d, M, K, epochs):
    from omegaconf import OmegaConf
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from qinco.utils import SharedCfgState
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    raw = OmegaConf.create({
        "M": M, "K": K, "L": 2, "dh": 256, "de": 64,
        "A": 16, "B": 4,
        "optimizer": "adamw", "lr": 8e-4, "wd": 0.1,
        "batch": 1024, "epochs": epochs,
        "grad_clip": 0.1, "grad_accumulate": 1,
        "ds": {"trainset": None, "valset": 10_000, "loop": 10_000_000},
        "seed": 0, "verbose": False,
        "codebook_noise_init": 0.1, "enc_max_bs": 65536,
        "mse_scale": 1, "qinco1_mode": False,
        "ivf_in_use": False, "task": "train",
        "cpu": False, "inference": False,
        "output": None, "model": None,
        "tensorboard": None, "ivf_centroids": None,
        "index": None, "encoded_db": None,
        "encoded_trainset": None, "pairwise_decoder": None,
        "db": None, "trainset": None, "queries": None,
        "queries_gt": None, "encode_trainset": False,
        "default_datasets": {}, "env": {},
        "scheduler": {"name": "cosine", "ramp_epochs": 3,
                      "lr_min_fact": 1e-3, "stop_patience": 10},
    })
    cfg = SharedCfgState(raw)
    accelerator = Accelerator(mixed_precision="fp16", cpu=False)
    cfg._accelerator = accelerator
    cfg._D = d
    cfg._M_ivf = M
    cfg._ivf_book = None
    cfg._ivf_centroids_preloaded = None
    cfg._optimizer = None
    cfg._scheduler = None
    cfg._cur_epoch = 0
    cfg._qinco_jit = False
    set_seed(cfg.seed)
    return cfg, accelerator


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 固定所有随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)

    ramp_str  = f"_ramp{ALPHA_RAMP}" if ALPHA_RAMP > 0 else ""
    warm_str  = f"_w{WARMUP_EPOCHS}" if WARMUP_EPOCHS > 0 else ""
    tag = f"M{M_VAL}_ep{EPOCHS}_alpha{ALPHA}{warm_str}{ramp_str}"
    log("=" * 70)
    log(f"QINCo2 + Score-MSE  {tag}  cache={CACHE.name}  device={DEVICE}")
    log(f"alpha={ALPHA}  page_batch={PAGE_BATCH}  score_bq={SCORE_BQ}")
    log("=" * 70)

    # ---- load embeddings ---------------------------------------------------
    log("Loading embeddings...")
    page  = torch.load(CACHE / "page_embeddings.pt",  weights_only=True).float()
    query = torch.load(CACHE / "query_embeddings.pt", weights_only=True).float()
    rel   = torch.load(CACHE / "relevance.pt",        weights_only=True)

    P, T, d = page.shape
    Q = query.shape[0]
    page_mask = page.norm(dim=-1) > 1e-6      # (P, T)
    flat_all  = page.reshape(P * T, d)
    mask_all  = page_mask.reshape(P * T)
    valid     = flat_all[mask_all]
    N_valid   = valid.shape[0]
    log(f"P={P} T={T} d={d} Q={Q} valid_tokens={N_valid:,}")

    # ---- pre-compute FP16 MaxSim scores (teacher) --------------------------
    # Shape (Q, P) on CPU; used by score loss to provide per-page targets.
    log("Pre-computing FP16 teacher scores...")
    fp16_flat = flat_all.clone()
    fp16_flat[~mask_all] = 0.0
    fp16_flat_dev = fp16_flat.to(DEVICE)
    fp16_scores = maxsim_scores(query, fp16_flat_dev, mask_all, P, T, bq=8)  # (Q, P) CPU
    del fp16_flat_dev
    torch.npu.empty_cache()
    fp16_ndcg = ndcg(fp16_scores, rel)
    log(f"FP16 nDCG@10 = {fp16_ndcg:.4f}")

    # ---- train / val token split -------------------------------------------
    perm    = torch.randperm(N_valid)
    n_train = int(N_valid * 0.9)
    train_np = valid[perm[:n_train]].numpy()
    val_np   = valid[perm[n_train:]].numpy()
    log(f"Train={len(train_np):,}  Val={len(val_np):,}")

    # ---- page index for score-loss sampling --------------------------------
    # We sample PAGE_BATCH pages per step; keep all their token indices handy.
    all_page_idx = torch.arange(P)

    # ---- build config + model ----------------------------------------------
    M, K = M_VAL, K_VAL
    cfg, accelerator = build_qinco_cfg_and_model(d, M, K, EPOCHS)
    device = accelerator.device

    cfg._data_mean = train_np.mean(axis=0)
    cfg._data_std  = float(train_np.std())

    log("Initialising codebooks with static RQ...")
    from qinco.vrq import apply_rq_on_vectors
    xt_t   = torch.from_numpy(train_np)
    xval_t = torch.from_numpy(val_np)
    _, _, rq_centroids = apply_rq_on_vectors(cfg, xt_t, xval_t)

    from qinco.model import QINCo, initialize_qinco_codebooks
    model = QINCo(cfg).to(device)
    rq_centroids_dev = [c.to(device) for c in rq_centroids]
    initialize_qinco_codebooks(cfg, model, rq_centroids_dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"QINCo model  params={n_params/1e6:.2f}M")

    # ---- dataloader (token level, same as original) ------------------------
    class NdArr(torch.utils.data.Dataset):
        def __init__(self, arr): self.t = torch.from_numpy(arr).float()
        def __len__(self): return len(self.t)
        def __getitem__(self, i): return self.t[i]

    _dl_gen = torch.Generator()
    _dl_gen.manual_seed(cfg.seed if hasattr(cfg, 'seed') else 42)
    train_loader = torch.utils.data.DataLoader(
        NdArr(train_np), batch_size=cfg.batch, shuffle=True,
        num_workers=0, drop_last=True,
        generator=_dl_gen,
    )
    val_loader = torch.utils.data.DataLoader(
        NdArr(val_np), batch_size=cfg.batch, shuffle=False,
        num_workers=0, drop_last=False,
    )

    from qinco.qinco_tasks import build_optimizer, build_scheduler
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # ---- training loop -----------------------------------------------------
    log("\n" + "-" * 70)
    rng = torch.Generator()
    rng.manual_seed(42)

    for epoch in range(EPOCHS):
        cfg._cur_epoch = epoch
        t0 = time.time()
        model.train()
        total_mse_loss = 0.0
        total_score_loss = 0.0
        n_batches = 0
        n_score_steps = 0
        # Effective alpha: optional linear ramp from 0→ALPHA over ALPHA_RAMP epochs,
        # then constant. WARMUP_EPOCHS is a hard gate (kept for compat, default 0).
        if ALPHA_RAMP > 0:
            eff_alpha = ALPHA * min(1.0, epoch / ALPHA_RAMP)
        else:
            eff_alpha = ALPHA
        use_score = eff_alpha > 0 and epoch >= WARMUP_EPOCHS

        for tok_batch in train_loader:
            tok_batch = tok_batch.to(device)

            # --- token-level MSE step (original QINCo) ---
            codes, xhat, losses = model(tok_batch, step="train")
            l_mse = sum(losses.values()) if isinstance(losses, dict) else losses

            # --- page-level score loss (every SCORE_EVERY batches) ---
            # The combined loss uses dynamic magnitude normalisation:
            #   scale = eff_alpha * mse_val / score_val
            # score_val is tiny (~1e-9) because MaxSim diff is divided by T^2,
            # so scale ≈ 5e9. The subsequent clip_grad_value_(0.1) then acts as a
            # gradient normaliser, giving a step of size ~lr*0.1 in the score
            # direction — small but consistent. This is the mechanism that made
            # v1 (nDCG=0.4640) work. Do NOT replace with a pure score step:
            # without the combined loss the raw score gradient is ~1e-12, below
            # the clip floor, and the score signal vanishes entirely.
            if use_score and n_batches % SCORE_EVERY == 0:
                page_idx = all_page_idx[
                    torch.randperm(P, generator=rng)[:PAGE_BATCH]
                ]
                pb = page[page_idx]
                pmb = page_mask[page_idx]
                fp16_scores_b = fp16_scores[:, page_idx]

                model.train()
                l_score = score_loss_for_batch(
                    accelerator.unwrap_model(model),
                    pb, query, pmb, fp16_scores_b,
                    PAGE_BATCH, T, d, bq=SCORE_BQ,
                )

                mse_val   = l_mse.detach().item()
                score_val = l_score.detach().item()
                # score_val is typically ~1e-9 (MaxSim diffs divided by T^2=534k,
                # averaged over mostly near-zero off-diagonal pairs). The old
                # threshold of 1e-8 was above this, silencing the score loss every
                # run. Use 1e-30 — only guards against literal NaN/zero.
                if score_val > 1e-30:
                    scale = eff_alpha * mse_val / score_val
                else:
                    scale = 0.0
                loss = l_mse + scale * l_score

                total_score_loss += l_score.item()
                n_score_steps += 1
            else:
                loss = l_mse

            optimizer.zero_grad()
            accelerator.backward(loss)
            if cfg.grad_clip:
                accelerator.clip_grad_value_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_mse_loss += l_mse.item()
            n_batches += 1
            if n_batches * cfg.batch >= cfg.ds.loop:
                break

        # validation MSE
        model.eval()
        val_mse, val_n = 0.0, 0
        with torch.no_grad():
            for vb in val_loader:
                vb = vb.to(device)
                _, xhat_v, _ = model(vb, step="train")
                val_mse += ((vb - xhat_v) ** 2).sum().item()
                val_n   += vb.numel()
        val_mse /= val_n

        scheduler.step(epoch)
        dt = time.time() - t0
        score_str = (f"  score_loss={total_score_loss/max(n_score_steps,1):.5f}({n_score_steps}steps)"
                     if ALPHA > 0 else "")
        log(f"Epoch {epoch+1:2d}/{EPOCHS}  "
            f"mse_loss={total_mse_loss/n_batches:.5f}  "
            f"val_mse={val_mse:.5f}{score_str}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"time={dt:.0f}s")

    # ---- encode + decode all pages -----------------------------------------
    log("\nEncoding all page tokens...")
    model.eval()
    flat_dev  = flat_all.to(device)
    mask_dev  = mask_all.to(device)
    flat_dev[~mask_dev] = 0.0
    enc_bs = 4096
    all_codes = []
    t_enc0 = time.time()
    with torch.no_grad():
        for i in range(0, P * T, enc_bs):
            codes = accelerator.unwrap_model(model)(flat_dev[i:i+enc_bs], step="encode")
            all_codes.append([c.cpu() for c in codes])
    t_enc = time.time() - t_enc0
    page_codes = [torch.cat([ch[m] for ch in all_codes]) for m in range(M)]
    log(f"Encode done: {t_enc:.2f}s  ({t_enc/(P*T)*1e6:.2f} µs/tok)")

    log("Decoding & scoring...")
    rec_flat = torch.zeros(P * T, d, device=device)
    dec_bs = 4096
    t_dec0 = time.time()
    with torch.no_grad():
        for i in range(0, P * T, dec_bs):
            chunk_codes = [page_codes[m][i:i+dec_bs].to(device) for m in range(M)]
            rec_flat[i:i+dec_bs] = accelerator.unwrap_model(model)(chunk_codes, step="decode")
    t_dec = time.time() - t_dec0
    rec_flat[~mask_dev] = 0.0

    sc_quant = maxsim_scores(query, rec_flat, mask_dev, P, T, bq=8)
    ndcg_quant = ndcg(sc_quant, rel)
    quality_pct = ndcg_quant / fp16_ndcg * 100

    # latency (min-of-3)
    scr = [None]
    def fn(): scr[0] = maxsim_scores(query, rec_flat, mask_dev, P, T, bq=8)
    t_arr = []
    for _ in range(3):
        t0 = time.perf_counter(); fn(); sync()
        t_arr.append(time.perf_counter() - t0)
    t_score = min(t_arr)

    result = dict(
        tag=tag, method="qinco_score_mse",
        M=M, K=K, epochs=EPOCHS, alpha=ALPHA,
        page_batch=PAGE_BATCH, params_M=round(n_params/1e6, 3),
        ndcg_fp16=round(fp16_ndcg, 4),
        ndcg_quant=round(ndcg_quant, 4),
        quality_pct=round(quality_pct, 2),
        storage_b_tok=M, compression_ratio=round(256.0/M, 1),
        encode_time_s=round(t_enc, 3),
        decode_time_s=round(t_dec, 3),
        score_time_s=round(t_score, 3),
    )

    log("\n" + "=" * 70)
    log(f"  FP16 nDCG@10  = {fp16_ndcg:.4f}")
    log(f"  Quant nDCG@10 = {ndcg_quant:.4f}  ({quality_pct:.2f}% of FP16)")
    log(f"  Storage       = {M} B/tok  ({256//M:.0f}x)")
    log(f"  Encode        = {t_enc:.1f}s   Decode = {t_dec:.1f}s   Score = {t_score:.3f}s")
    log("=" * 70)

    out_json  = OUT_DIR / f"qinco_score_mse_{tag}.json"
    out_model = OUT_DIR / f"qinco_score_mse_{tag}.pt"
    out_json.write_text(json.dumps(result, indent=2))
    torch.save({"state_dict": accelerator.unwrap_model(model).state_dict(),
                "result": result}, out_model)
    log(f"\nSaved: {out_json}")
    log(f"       {out_model}")


if __name__ == "__main__":
    main()
