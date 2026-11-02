#!/usr/bin/env python3
"""QINCo V2 M=16 纯MSE baseline (alpha=0.0)."""
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
    raise RuntimeError(f"repo root not found above {p}")

ROOT = _find_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiment/methods/token_quantization/qinco2"))

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

DEVICE = os.environ.get("DEVICE", "npu:0")
SEED = int(os.environ.get("SEED", 42))
EPOCHS = 10
M_VAL = 16
K_VAL = 256
CACHE = Path(os.environ.get("CACHE_DIR",
    str(ROOT / "experiment/embeddings/colqwen2.5_v3/physics")))
OUT = Path(os.environ.get("OUT_DIR",
    str(ROOT / "experiment/results/quant")))
OUT.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT / f"qinco_mse_{CACHE.name}.json"

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

def build_qinco(d, M, K, epochs):
    from omegaconf import OmegaConf
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from qinco.utils import SharedCfgState
    from torch_npu.contrib import transfer_to_npu
    raw = OmegaConf.create({
        "M": M, "K": K, "L": 2, "dh": 256, "de": 64, "A": 16, "B": 4,
        "optimizer": "adamw", "lr": 8e-4, "wd": 0.1, "batch": 1024, "epochs": epochs,
        "grad_clip": 0.1, "grad_accumulate": 1,
        "ds": {"trainset": None, "valset": 10_000, "loop": 10_000_000},
        "seed": 0, "verbose": False, "codebook_noise_init": 0.1,
        "enc_max_bs": 65536, "mse_scale": 1, "qinco1_mode": False,
        "ivf_in_use": False, "task": "train", "cpu": False, "inference": False,
        "output": None, "model": None, "tensorboard": None,
        "ivf_centroids": None, "index": None, "encoded_db": None,
        "encoded_trainset": None, "pairwise_decoder": None,
        "db": None, "trainset": None, "queries": None, "queries_gt": None,
        "encode_trainset": False, "default_datasets": {}, "env": {},
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

log("=" * 70)
log(f"QINCo V2 M=16 纯MSE (alpha=0.0)  epochs={EPOCHS}")
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

log(f"P={P} T={T} d={d} Q={Q} valid={valid.shape[0]:,}")

perm = torch.randperm(valid.shape[0], generator=torch.Generator().manual_seed(42))
n_train = int(valid.shape[0] * 0.9)
train_np = valid[perm[:n_train]].numpy()
val_np = valid[perm[n_train:]].numpy()

M, K = M_VAL, K_VAL
cfg, accelerator = build_qinco(d, M, K, EPOCHS)
device = accelerator.device
cfg._data_mean = train_np.mean(axis=0)
cfg._data_std = float(train_np.std())

log("Initialising codebooks with static RQ...")
from qinco.vrq import apply_rq_on_vectors
from qinco.model import QINCo, initialize_qinco_codebooks
xt_t = torch.from_numpy(train_np)
xval_t = torch.from_numpy(val_np)
_, _, rq_centroids = apply_rq_on_vectors(cfg, xt_t, xval_t)
model = QINCo(cfg).to(device)
rq_centroids_dev = [c.to(device) for c in rq_centroids]
initialize_qinco_codebooks(cfg, model, rq_centroids_dev)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"Model params={n_params/1e6:.2f}M")

class NdArr(torch.utils.data.Dataset):
    def __init__(self, arr): self.t = torch.from_numpy(arr).float()
    def __len__(self): return len(self.t)
    def __getitem__(self, i): return self.t[i]

_dl_gen = torch.Generator()
_dl_gen.manual_seed(SEED)
train_loader = torch.utils.data.DataLoader(
    NdArr(train_np), batch_size=cfg.batch, shuffle=True,
    num_workers=0, drop_last=True, generator=_dl_gen)
val_loader = torch.utils.data.DataLoader(
    NdArr(val_np), batch_size=cfg.batch, shuffle=False,
    num_workers=0, drop_last=False)

from qinco.qinco_tasks import build_optimizer, build_scheduler
optimizer = build_optimizer(cfg, model)
scheduler = build_scheduler(cfg, optimizer)
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

log("\n" + "-" * 70)
t_train0 = time.time()
for epoch in range(EPOCHS):
    cfg._cur_epoch = epoch
    t0 = time.time()
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in train_loader:
        batch = batch.to(device)
        codes, xhat, losses = model(batch, step="train")
        loss = sum(losses.values()) if isinstance(losses, dict) else losses
        optimizer.zero_grad()
        accelerator.backward(loss)
        if cfg.grad_clip:
            accelerator.clip_grad_value_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        if n_batches * cfg.batch >= cfg.ds.loop:
            break
    model.eval()
    val_mse, val_n = 0.0, 0
    with torch.no_grad():
        for vb in val_loader:
            vb = vb.to(device)
            _, xhat_v, _ = model(vb, step="train")
            val_mse += ((vb - xhat_v) ** 2).sum().item()
            val_n += vb.numel()
    val_mse /= val_n
    scheduler.step(epoch)
    dt = time.time() - t0
    log(f"Epoch {epoch+1:2d}/{EPOCHS}  loss={total_loss/n_batches:.5f}  "
        f"val_mse={val_mse:.5f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
        f"time={dt:.0f}s")
t_train = time.time() - t_train0

log("\nEncoding...")
model.eval()
flat_npu = flat_all.to(device)
mask_npu = mask_all.to(device)
flat_npu[~mask_npu] = 0.0
enc_bs = 4096
all_codes = []
t_enc0 = time.time()
with torch.no_grad():
    for i in range(0, P*T, enc_bs):
        codes = accelerator.unwrap_model(model)(flat_npu[i:i+enc_bs], step="encode")
        all_codes.append([c.cpu() for c in codes])
t_enc = time.time() - t_enc0
page_codes = [torch.cat([ch[m] for ch in all_codes]) for m in range(M)]

log("Decoding...")
rec_flat = torch.zeros(P*T, d, device=device)
dec_bs = 4096
t_dec0 = time.time()
with torch.no_grad():
    for i in range(0, P*T, dec_bs):
        chunk_codes = [page_codes[m][i:i+dec_bs].to(device) for m in range(M)]
        rec_flat[i:i+dec_bs] = accelerator.unwrap_model(model)(chunk_codes, step="decode")
t_dec = time.time() - t_dec0
rec_flat[~mask_npu] = 0.0

fp16_flat = flat_all.to(device)
fp16_flat[~mask_npu] = 0.0
sc_fp16 = maxsim(query, fp16_flat, mask_all, P, T)
sc_qinco = maxsim(query, rec_flat, mask_all, P, T)
ndcg_fp16 = ndcg(sc_fp16, rel)
ndcg_qinco = ndcg(sc_qinco, rel)
quality_pct = ndcg_qinco / ndcg_fp16 * 100

box = [None]
def fn(): box[0] = maxsim(query, rec_flat, mask_all, P, T)
t_score = timed(fn)

kb = avg_tok * M / 1024

result = dict(
    method="qinco_mse",
    M=M, K=K, epochs=EPOCHS, alpha=0.0,
    params_M=round(n_params/1e6, 3),
    ndcg_fp16=round(ndcg_fp16, 4),
    ndcg_qinco=round(ndcg_qinco, 4),
    quality_pct=round(quality_pct, 2),
    storage_kb_per_page=round(kb, 1),
    train_time_s=round(t_train, 1),
    encode_time_s=round(t_enc, 3),
    decode_time_s=round(t_dec, 3),
    score_time_s=round(t_score, 3),
)

log(f"\nResult: nDCG@10 = {ndcg_qinco:.4f} ({quality_pct:.2f}% of FP16)")
log(f"  Train={t_train:.1f}s  Encode={t_enc:.2f}s  Decode={t_dec:.2f}s  Score={t_score:.3f}s")
OUT_FILE.write_text(json.dumps(result, indent=2))
log(f"Saved: {OUT_FILE}")
