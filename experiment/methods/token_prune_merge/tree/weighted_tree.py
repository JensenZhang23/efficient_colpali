"""
Importance-weighted tree construction for efficient ColPali retrieval.

Extends the baseline uniform k-means tree by using token importance weights
to build better clusters that preserve high-information regions.
"""
import torch
import numpy as np
from typing import Literal


def compute_token_importance(
    embeddings: torch.Tensor,
    importance_type: Literal["uniform", "variance", "norm"]
) -> torch.Tensor:
    """
    Compute importance weight for each token.

    Args:
        embeddings: [num_tokens, dim]
        importance_type: weighting scheme
            - "uniform": all tokens equal weight (baseline)
            - "variance": use embedding variance across dimensions
            - "norm": use L2 norm of embedding

    Returns:
        weights: [num_tokens] importance scores
    """
    if importance_type == "uniform":
        return torch.ones(embeddings.shape[0], device=embeddings.device)

    elif importance_type == "variance":
        # Higher variance = more information
        var = embeddings.var(dim=-1)
        return var

    elif importance_type == "norm":
        # Higher norm = more salient
        norm = embeddings.norm(dim=-1)
        return norm

    elif importance_type == "combined":
        # Combined: 0.6*norm + 0.4*variance
        norm_score = embeddings.norm(dim=-1)
        var_score = embeddings.float().var(dim=-1)
        # Normalize to [0, 1]
        norm_score = (norm_score - norm_score.min()) / (norm_score.max() - norm_score.min() + 1e-8)
        var_score = (var_score - var_score.min()) / (var_score.max() - var_score.min() + 1e-8)
        return 0.6 * norm_score + 0.4 * var_score

    else:
        raise ValueError(f"Unknown importance_type: {importance_type}")


def weighted_kmeans(
    X: torch.Tensor,
    weights: torch.Tensor,
    n_clusters: int,
    max_iter: int = 3
) -> torch.Tensor:
    """
    Fast weighted sampling instead of full k-means.

    Simply samples n_clusters tokens proportional to their importance weights.
    Much faster than iterative k-means for large-scale compression.

    Args:
        X: [N, dim] input vectors
        weights: [N] importance weights
        n_clusters: number of tokens to sample
        max_iter: ignored (kept for API compatibility)

    Returns:
        sampled: [n_clusters, dim] sampled tokens
    """
    # Sample proportional to importance weights
    probs = weights / weights.sum()
    indices = torch.multinomial(probs, n_clusters, replacement=False)
    return X[indices].clone()


def weighted_topk(
    X: torch.Tensor,
    weights: torch.Tensor,
    n_clusters: int
) -> torch.Tensor:
    """
    Deterministic top-k selection by importance weight.

    Unlike weighted_kmeans (probabilistic sampling), this GUARANTEES the
    n_clusters most important tokens are retained. Addresses the failure mode
    at low budgets where important tokens may be missed by random sampling.

    Args:
        X: [N, dim] input vectors
        weights: [N] importance weights
        n_clusters: number of tokens to select

    Returns:
        selected: [n_clusters, dim] top-k tokens by importance
    """
    _, indices = torch.topk(weights, n_clusters)
    return X[indices].clone()


def build_importance_weighted_tree(
    page_embeddings: torch.Tensor,
    budget: int,
    importance_type: Literal["variance", "norm"] = "variance",
    device: str = "cpu",
    strategy: Literal["sample", "topk"] = "topk"
) -> torch.Tensor:
    """
    Build importance-weighted tree index for all pages.

    Args:
        page_embeddings: [num_pages, max_tokens, dim]
        budget: target tokens per page
        importance_type: how to compute token importance
        device: computation device
        strategy: token selection strategy
            - "topk": deterministic top-k by importance (guarantees retention)
            - "sample": probabilistic weighted sampling

    Returns:
        tree_pages: [num_pages, budget, dim] compressed embeddings
    """
    num_pages = page_embeddings.shape[0]
    dim = page_embeddings.shape[2]

    page_embeddings = page_embeddings.to(device)
    tree_pages = torch.zeros(num_pages, budget, dim, device=device, dtype=page_embeddings.dtype)

    for i in range(num_pages):
        page = page_embeddings[i]  # [max_tokens, dim]

        # Filter padding
        valid_mask = page.norm(dim=-1) > 1e-6
        valid_tokens = page[valid_mask]

        if valid_tokens.shape[0] == 0:
            continue

        # Compute importance weights
        importance = compute_token_importance(valid_tokens, importance_type)

        # Build weighted tree
        n_clusters = min(budget, valid_tokens.shape[0])
        if n_clusters < valid_tokens.shape[0]:
            if strategy == "topk":
                centroids = weighted_topk(valid_tokens, importance, n_clusters)
            else:
                centroids = weighted_kmeans(valid_tokens, importance, n_clusters)
        else:
            centroids = valid_tokens

        tree_pages[i, :centroids.shape[0]] = centroids

    return tree_pages


def build_uniform_tree(
    page_embeddings: torch.Tensor,
    budget: int,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Baseline: uniform k-means tree (no importance weighting).

    Args:
        page_embeddings: [num_pages, max_tokens, dim]
        budget: target tokens per page
        device: computation device

    Returns:
        tree_pages: [num_pages, budget, dim] compressed embeddings
    """
    num_pages = page_embeddings.shape[0]
    dim = page_embeddings.shape[2]

    page_embeddings = page_embeddings.to(device)
    tree_pages = torch.zeros(num_pages, budget, dim, device=device, dtype=page_embeddings.dtype)

    for i in range(num_pages):
        page = page_embeddings[i]
        valid_mask = page.norm(dim=-1) > 1e-6
        valid_tokens = page[valid_mask]

        if valid_tokens.shape[0] == 0:
            continue

        # Uniform weights
        weights = torch.ones(valid_tokens.shape[0], device=device)

        n_clusters = min(budget, valid_tokens.shape[0])
        if n_clusters < valid_tokens.shape[0]:
            centroids = weighted_kmeans(valid_tokens, weights, n_clusters)
        else:
            centroids = valid_tokens

        tree_pages[i, :centroids.shape[0]] = centroids

    return tree_pages


def evaluate_tree(
    tree_pages: torch.Tensor,
    query_embeddings: torch.Tensor,
    relevance: torch.Tensor,
    device: str,
    batch_size: int = 256,
    k: int = 5,
) -> dict:
    """
    Evaluate tree index with MaxSim scoring.

    Args:
        tree_pages: [num_pages, budget, dim]
        query_embeddings: [num_queries, max_query_len, dim]
        relevance: [num_queries, num_pages] ground truth
        device: computation device
        batch_size: number of queries to process at once
        k: NDCG cutoff (5 for legacy, 10 to match the ViDoRe V3 paper)

    Returns:
        dict with ndcg (NDCG@k), ndcg_at_5 (alias for backward compat),
        time_sec, memory_MB
    """
    import time

    # Inline ndcg_at_k to avoid dependency on leaf_exact_tree
    def ndcg_at_k(rankings, relevance):
        """Compute NDCG@k from rankings and relevance matrix.

        Args:
            rankings: (Q, k) int tensor of ranked document indices
            relevance: (Q, P) float tensor of relevance scores

        Returns:
            (Q,) float tensor of NDCG@k scores per query
        """
        Q, k = rankings.shape
        gains = relevance.gather(1, rankings)  # (Q, k)
        discounts = torch.log2(torch.arange(2, k + 2, dtype=torch.float32))  # (k,)
        dcg = (gains / discounts).sum(dim=1)  # (Q,)

        # Ideal DCG: sort relevance in descending order
        ideal_gains, _ = relevance.topk(k, dim=1)  # (Q, k)
        idcg = (ideal_gains / discounts).sum(dim=1)  # (Q,)

        ndcg = dcg / idcg.clamp_min(1e-10)  # avoid division by zero
        return ndcg

    start = time.time()

    query_embeddings = query_embeddings.to(device)
    tree_pages = tree_pages.to(device)

    # Normalize once
    query_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
    doc_norm = torch.nn.functional.normalize(tree_pages, p=2, dim=-1)

    query_mask = query_embeddings.norm(dim=-1) > 1e-6
    doc_mask = tree_pages.norm(dim=-1) > 1e-6

    # Compute scores in batches to avoid OOM
    num_queries = query_embeddings.shape[0]
    all_scores = []

    for i in range(0, num_queries, batch_size):
        end_i = min(i + batch_size, num_queries)
        batch_query_norm = query_norm[i:end_i]
        batch_query_mask = query_mask[i:end_i]

        # [batch, ql, d] @ [nd, dl, d] -> [batch, nd, ql, dl]
        scores_matrix = torch.einsum('nqd,mld->nmlq', batch_query_norm, doc_norm)
        scores_matrix = scores_matrix.permute(0, 1, 3, 2)

        query_mask_exp = batch_query_mask.unsqueeze(1).unsqueeze(3)
        doc_mask_exp = doc_mask.unsqueeze(0).unsqueeze(2)
        scores_matrix = scores_matrix * query_mask_exp.float() * doc_mask_exp.float()

        max_sims, _ = scores_matrix.max(dim=3)
        batch_scores = (max_sims * batch_query_mask.unsqueeze(1).float()).sum(dim=2)
        all_scores.append(batch_scores.cpu())

    scores = torch.cat(all_scores, dim=0)

    rankings = scores.argsort(dim=1, descending=True)[:, :k]
    ndcg_val = float(ndcg_at_k(rankings, relevance).mean())
    elapsed = time.time() - start

    # Memory estimate
    num_tokens_per_page = doc_mask.sum(dim=1).float().mean().item()
    memory_mb = (tree_pages.shape[0] * num_tokens_per_page * tree_pages.shape[2] * 4) / (1024**2)

    return {
        "ndcg": ndcg_val,
        "ndcg_at_5": ndcg_val,  # backward-compat alias (holds NDCG@k)
        "k": k,
        "time_sec": elapsed,
        "memory_MB": memory_mb,
    }
