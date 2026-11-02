"""ColBERTSaR: sparse approximation of the MaxSim score via product quantization.

Reference: Yang et al., "ColBERTSaR: Sparsified ColBERT Index via Product
Quantization", SIGIR '26 (arXiv:2606.05568).

Core idea
---------
Standard MaxSim scores a (query, doc) pair with the dense, residual-full form

    Score(q, d) = sum_i  max_j  (q_i . d_j)

which requires storing every document patch embedding as a float vector. This
module drops the residual entirely: each document patch d_j is replaced by the
id of its closest K-means anchor c_k, so a document becomes a *set of anchor
ids* v_d = { argmax_k (c_k . d_j) : d_j in d }. The score simplifies to

    Score_S(q, d) = sum_i  max_{k in v_d}  (q_i . c_k)                      (Eq. 3)

The query-to-anchor scores S = Q @ C^T (shape |q| x K) are computed once per
query. Retrieval then only needs, per document, the integer set v_d -- no
float embeddings. This turns the ColBERT index into a true inverted index:

    inverted[k]  -> list of doc ids whose v_d contains anchor k
    forward[d]   -> the anchor-id set v_d of document d

At search time: (1) take the top-nprobe anchors of each query token, union
their posting lists to form a candidate set; (2) rerank each candidate with
Eq. 3 using the forward index.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import torch

# Make colpali_engine importable whether we run from repo root or sparse/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from colpali_engine.utils.torch_utils import get_torch_device as _get_device
except ImportError:
    def _get_device(device: str = "auto") -> str:  # minimal fallback
        try:
            import torch_npu  # noqa: F401
            if torch.npu.is_available():
                return "npu:0"
        except ImportError:
            pass
        return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_device(device) -> torch.device:
    """Accept str | torch.device | None and always return torch.device."""
    if device is None:
        return torch.device(_get_device("auto"))
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


@torch.no_grad()
def kmeans_fit(
    x: torch.Tensor,
    k: int,
    n_iter: int = 25,
    seed: int = 0,
    tol: float = 1e-4,
    verbose: bool = False,
) -> torch.Tensor:
    """Fit k centroids on rows of x with Lloyd's algorithm (cosine/IP).

    Embeddings are assumed L2-normalized, so inner product == cosine similarity
    and "closest anchor" == "highest dot product". Runs on whatever device x
    is on -- pass an NPU/CUDA tensor to fit on accelerator.

    Returns:
        (k, D) centroid matrix on the same device as x.
    """
    x = _l2_normalize(x.float())
    n, d = x.shape
    if k > n:
        raise ValueError(f"k={k} exceeds the number of training points N={n}")

    g = torch.Generator(device="cpu").manual_seed(seed)
    init_idx = torch.randperm(n, generator=g)[:k]
    centroids = x[init_idx].clone()

    prev_assign = None
    for it in range(n_iter):
        # Assign: argmax dot-product in row-chunks to bound peak memory.
        assign = torch.empty(n, dtype=torch.long, device=x.device)
        chunk = max(1, 1_000_000 // max(1, k))
        for start in range(0, n, chunk):
            sims = x[start : start + chunk] @ centroids.T
            assign[start : start + chunk] = sims.argmax(dim=1)

        # Update: mean of assigned points, renormalized; empty clusters re-seeded.
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=x.device)
        new_centroids.index_add_(0, assign, x)
        counts.index_add_(0, assign, torch.ones(n, device=x.device))
        empty = counts == 0
        new_centroids = new_centroids / counts.clamp_min(1.0).unsqueeze(1)
        if empty.any():
            n_empty = int(empty.sum().item())
            reseed = torch.randperm(n, generator=g)[:n_empty]
            new_centroids[empty] = x[reseed]
        centroids = _l2_normalize(new_centroids)

        if prev_assign is not None:
            moved = (assign != prev_assign).float().mean().item()
            if verbose:
                print(f"[kmeans] iter {it+1}/{n_iter}  reassigned={moved:.4f}", flush=True)
            if moved < tol:
                break
        prev_assign = assign

    return centroids


@dataclass
class ColBERTSaRIndex:
    """Residual-free sparse ColBERT index.

    Attributes:
        centroids: (K, D) anchor matrix, L2-normalized float32 on CPU.
        forward:   list[i] = 1-D LongTensor of unique anchor ids in doc i (v_d).
        inverted:  list[k] = 1-D LongTensor of doc ids whose v_d contains anchor k.
        doc_ids:   optional external id strings aligned with 0..n_docs-1.
    """

    centroids: torch.Tensor
    forward: List[torch.Tensor] = field(default_factory=list)
    inverted: List[torch.Tensor] = field(default_factory=list)
    doc_ids: Optional[List[str]] = None

    @property
    def k(self) -> int:
        return self.centroids.shape[0]

    @property
    def n_docs(self) -> int:
        return len(self.forward)

    # ------------------------------------------------------------------ build

    @classmethod
    @torch.no_grad()
    def build(
        cls,
        doc_embeddings: Sequence[torch.Tensor],
        k: int,
        *,
        sample_size: Optional[int] = 500_000,
        n_iter: int = 25,
        device=None,
        doc_ids: Optional[List[str]] = None,
        seed: int = 0,
        verbose: bool = True,
    ) -> "ColBERTSaRIndex":
        """Fit anchors and build inverted + forward index.

        Args:
            doc_embeddings: list of (n_patches_i, D) per-document embeddings.
            k: number of anchors.
            sample_size: cap on patches used to fit K-means (None = all).
            n_iter: K-means iterations.
            device: str, torch.device, or None (auto-detect NPU/CUDA/CPU).
            doc_ids: optional external id strings aligned with the list.
        """
        dev = _resolve_device(device)

        all_patches = torch.cat([e.float() for e in doc_embeddings], dim=0)
        n_total = all_patches.shape[0]
        if sample_size is not None and n_total > sample_size:
            g = torch.Generator().manual_seed(seed)
            sel = torch.randperm(n_total, generator=g)[:sample_size]
            sample = all_patches[sel]
        else:
            sample = all_patches

        if verbose:
            print(
                f"[colbertsar] fitting {k} anchors on {sample.shape[0]}/{n_total} "
                f"patches (dim={sample.shape[1]}) on {dev}",
                flush=True,
            )
        t0 = time.time()
        centroids = kmeans_fit(sample.to(dev), k=k, n_iter=n_iter, seed=seed, verbose=verbose)
        if verbose:
            print(f"[colbertsar] k-means done in {time.time()-t0:.1f}s", flush=True)

        # Assign every patch of every doc to its closest anchor.
        forward: List[torch.Tensor] = []
        posting: List[List[int]] = [[] for _ in range(k)]
        for d, emb in enumerate(doc_embeddings):
            e = _l2_normalize(emb.float()).to(dev)
            anchor_ids = (e @ centroids.T).argmax(dim=1)
            v_d = torch.unique(anchor_ids).cpu()
            forward.append(v_d)
            for a in v_d.tolist():
                posting[a].append(d)

        inverted = [torch.tensor(p, dtype=torch.long) for p in posting]
        if verbose:
            nnz = sum(v.numel() for v in forward)
            print(
                f"[colbertsar] indexed {len(forward)} docs, "
                f"avg |v_d|={nnz/max(1,len(forward)):.1f}, K={k}",
                flush=True,
            )
        return cls(
            centroids=centroids.cpu(),
            forward=forward,
            inverted=inverted,
            doc_ids=doc_ids,
        )

    # ------------------------------------------------------------------ score

    @torch.no_grad()
    def score(
        self,
        query_embeddings: Sequence[torch.Tensor],
        *,
        nprobe: int = 4,
        use_forward_index: bool = True,
        device=None,
        candidate_only: bool = False,
    ) -> torch.Tensor:
        """Score all documents for each query with the sparse Eq. 3 approximation.

        Args:
            query_embeddings: list of (n_qtokens_i, D) query embeddings.
            nprobe: top anchors explored per query token (first stage).
            use_forward_index: if True, second-stage exact Eq. 3 rerank;
                if False, first-stage-only (cheaper, less accurate).
            device: accelerator device for S = Q @ C^T.
            candidate_only: non-candidate docs get -inf when True.

        Returns:
            (n_queries, n_docs) float32 score tensor on CPU.
        """
        dev = _resolve_device(device)
        centroids = self.centroids.to(dev)
        scores = torch.full((len(query_embeddings), self.n_docs), float("-inf"))

        for qi, q in enumerate(query_embeddings):
            q = _l2_normalize(q.float()).to(dev)
            S = q @ centroids.T  # (|q|, K)

            top_anchors = S.topk(min(nprobe, self.k), dim=1).indices  # (|q|, nprobe)
            cand: set = set()
            for a in top_anchors.reshape(-1).tolist():
                post = self.inverted[a]
                if post.numel():
                    cand.update(post.tolist())

            if not cand:
                continue
            cand_list = sorted(cand)

            if use_forward_index:
                row = self._forward_score(S, cand_list, dev, restrict=None)
            else:
                row = self._forward_score(S, cand_list, dev, restrict=top_anchors)

            scores[qi, torch.tensor(cand_list)] = row.cpu()

        if not candidate_only:
            scores[scores == float("-inf")] = -1e30
        return scores.to(torch.float32)

    def _forward_score(
        self,
        S: torch.Tensor,
        cand_list: List[int],
        device: torch.device,
        restrict: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute Score_S(q, d) = sum_i max_{k in v_d} S[i,k] for each candidate."""
        probed_set = None
        if restrict is not None:
            probed_set = set(torch.unique(restrict).tolist())

        out = torch.empty(len(cand_list), device=device)
        for i, d in enumerate(cand_list):
            v_d = self.forward[d].to(device)
            if probed_set is not None:
                mask = torch.tensor(
                    [int(a) in probed_set for a in v_d.tolist()], device=device
                )
                if mask.any():
                    v_d = v_d[mask]
                else:
                    out[i] = -1e30
                    continue
            out[i] = S[:, v_d].max(dim=1).values.sum()
        return out

    # ------------------------------------------------------------------ utils

    def index_size_bytes(self) -> int:
        """Approximate footprint: centroids fp16 + forward posting ints."""
        return int(
            self.centroids.numel() * 2
            + sum(v.numel() for v in self.forward) * 4
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "centroids": self.centroids.half(),  # fp16 to save disk space
                "forward": self.forward,
                "inverted": self.inverted,
                "doc_ids": self.doc_ids,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "ColBERTSaRIndex":
        blob = torch.load(path, map_location=map_location)
        return cls(
            centroids=blob["centroids"].float(),
            forward=blob["forward"],
            inverted=blob["inverted"],
            doc_ids=blob.get("doc_ids"),
        )
