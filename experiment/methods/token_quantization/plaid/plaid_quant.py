#!/usr/bin/env python3
"""PLAID-style scalar quantization for ColBERT-style visual token compression.

PLAID (ColBERTv2) uses centroid-based residual compression with uniform scalar
quantization per dimension. Unlike RQ which learns M codebooks jointly on the
token distribution, PLAID quantization is data-independent: each dimension of
the residual is quantized uniformly into 2^B levels based on its min/max range.

This is the natural baseline for bit-level compression: RQ (learned codebooks)
vs PLAID (uniform scalar quantization). We expect RQ to win on visual tokens
due to higher redundancy and more structured embedding space.

Reference:
  Santhanam et al. "ColBERTv2: Effective and Efficient Retrieval via
  Lightweight Late Interaction over BERT." NeurIPS 2022.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def plaid_compress(page, bits_per_dim=2, device='cpu'):
    """PLAID-style compression: centroid + uniform scalar quantization.

    For each page independently:
      1. Compute centroid = mean(tokens)
      2. Compute residual = tokens - centroid
      3. Per dimension: find [min, max], quantize uniformly into 2^bits levels
      4. Reconstruct: dequantize residuals + add back centroid

    Args:
        page: (P, T, d) page embeddings (float, already L2-normalized by ColBERT)
        bits_per_dim: 2/4/8 bits per dimension for uniform quantization
        device: where to run computation

    Returns:
        (P, T, d) reconstructed page embeddings (same shape, float)
    """
    P, T, d = page.shape
    out = torch.zeros_like(page)
    levels = 2 ** bits_per_dim

    for pi in range(P):
        pg = page[pi].to(device)
        mask = pg.norm(dim=-1) > 1e-6  # non-padding tokens
        if not mask.any():
            out[pi] = pg.cpu()
            continue

        tokens = pg[mask]  # (n, d) where n = num real tokens

        # Step 1: Centroid (per-page "average" token, not stored per-token)
        centroid = tokens.mean(dim=0, keepdim=True)  # (1, d)

        # Step 2: Residuals
        residual = tokens - centroid  # (n, d)

        # Step 3: Per-dimension uniform scalar quantization
        # Each dimension independently: map [rmin, rmax] → [0, levels-1]
        rmin = residual.min(dim=0, keepdim=True).values  # (1, d)
        rmax = residual.max(dim=0, keepdim=True).values  # (1, d)
        scale = (rmax - rmin) / (levels - 1)
        scale = scale.clamp_min(1e-8)  # avoid division by zero

        # Quantize: float → int codes → dequantized float (simulate storage)
        codes = ((residual - rmin) / scale).round().clamp(0, levels - 1)
        residual_rec = codes * scale + rmin

        # Step 4: Reconstruct
        tokens_rec = residual_rec + centroid

        pg_out = pg.clone()
        pg_out[mask] = tokens_rec
        out[pi] = pg_out.cpu()

    return out


def bytes_per_token_plaid(bits_per_dim, d=128, avg_tokens_per_page=731):
    """Storage cost of PLAID compression.

    Per page:
      - 1 centroid: d × 2 bytes (float16)
      - T residual codes: T × d × bits_per_dim bits
      - Min/max per dimension: 2 × d × 2 bytes (float16, for dequantization)

    Amortized per token: (centroid_bytes + minmax_bytes + residual_bytes) / T

    Args:
        bits_per_dim: 2/4/8
        d: embedding dimension (default 128)
        avg_tokens_per_page: average non-padding tokens per page

    Returns:
        bytes per token (float)
    """
    centroid_bytes = d * 2  # float16
    minmax_bytes = 2 * d * 2  # min + max per dimension, float16
    residual_bytes = avg_tokens_per_page * d * bits_per_dim / 8
    total_bytes = centroid_bytes + minmax_bytes + residual_bytes
    return total_bytes / avg_tokens_per_page


@torch.no_grad()
def plaid_compress_batch(page, bits_per_dim=2, device='cpu', chunk_size=100):
    """Batched version of plaid_compress for large page tensors.

    Process pages in chunks to avoid OOM on device.
    """
    P = page.shape[0]
    results = []
    for start in range(0, P, chunk_size):
        end = min(start + chunk_size, P)
        chunk = page[start:end]
        results.append(plaid_compress(chunk, bits_per_dim=bits_per_dim, device=device))
    return torch.cat(results, dim=0)


# ── Comparison with RQ ────────────────────────────────────────────────────────

def compare_rq_vs_plaid():
    """Conceptual comparison (not executable without real data).

    RQ (Residual Quantization):
      - Learns M codebooks of size K=256 from data
      - Greedy residual encoding: stage m approximates residual from stage m-1
      - Storage: M × log2(K) bits per token (M bytes when K=256)
      - Advantage: data-driven, captures token distribution structure

    PLAID (Uniform Scalar Quantization):
      - Per-page centroid + per-dimension uniform quantization
      - Storage: (d×2 + T×d×B/8) / T bytes per token, where B = bits_per_dim
      - For typical T=731, B=2: ~2.35 bytes/token
      - Advantage: simple, no codebook training, memory-efficient

    Hypothesis for visual tokens:
      - Visual document tokens have higher redundancy (background, layout)
      - Embedding space is more structured (adjacent patches correlate)
      - RQ's learned codebooks should capture this structure better
      - At same bit budget: RQ M=2 (2 bytes/tok) > PLAID 2-bit (2.35 bytes/tok)
    """
    pass


if __name__ == "__main__":
    # Smoke test
    P, T, d = 10, 731, 128
    page = torch.randn(P, T, d)
    page = page / page.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # L2 normalize

    for bits in [2, 4, 8]:
        rec = plaid_compress(page, bits_per_dim=bits, device='cpu')
        mse = ((page - rec) ** 2).mean().item()
        bytes_tok = bytes_per_token_plaid(bits, d=d, avg_tokens_per_page=T)
        print(f"PLAID {bits}-bit: MSE={mse:.6f}  bytes/token={bytes_tok:.2f}")
