#!/usr/bin/env python3
"""
PRUNE-THEN-MERGE benchmark on physics dataset.
Reuses cached embeddings from tree_index_exp/bench_all/physics/.
"""
import sys
import json
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from prune_merge.prune_then_merge import prune_then_merge, maxsim_scores
from scripts.run_bench_all import ndcg5, _discover_datasets, _read_arrow, build_relevance


def load_dataset(dataset_name: str):
    """Load dataset metadata (corpus, queries, relevance)."""
    datasets = _discover_datasets()

    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not found")

    config = datasets[dataset_name]
    dataset_root = Path(config["root"])

    # Read corpus
    corpus = _read_arrow(dataset_root, "corpus")

    # Read queries
    queries = _read_arrow(dataset_root, "queries")

    # Read qrels
    qrels_raw = _read_arrow(dataset_root, "qrels")

    # Extract IDs (V3 uses underscore, V2 uses hyphen)
    query_id_key = "query_id" if "query_id" in queries[0] else "query-id"
    corpus_id_key = "corpus_id" if "corpus_id" in corpus[0] else "corpus-id"

    query_ids = [q[query_id_key] for q in queries]
    corpus_ids = [c[corpus_id_key] for c in corpus]

    # Build relevance matrix
    relevance = build_relevance(qrels_raw, query_ids, corpus_ids, qid_key=query_id_key, cid_key=corpus_id_key)

    return corpus, queries, relevance, config.get("vidore_version", "v3")


def load_cached_embeddings(dataset_name: str):
    """Load cached embeddings and relevance from previous tree benchmark."""
    embed_dir = Path("tree_index_exp/bench_all") / dataset_name

    page_emb_file = embed_dir / "page_embeddings.pt"
    query_emb_file = embed_dir / "query_embeddings.pt"
    relevance_file = embed_dir / "relevance.pt"

    if not page_emb_file.exists() or not query_emb_file.exists() or not relevance_file.exists():
        raise FileNotFoundError(
            f"Cached embeddings not found in {embed_dir}.\n"
            f"Run: python scripts/run_bench_all.py --datasets {dataset_name} first."
        )

    page_embeddings = torch.load(page_emb_file, weights_only=True)
    query_embeddings = torch.load(query_emb_file, weights_only=True)
    relevance = torch.load(relevance_file, weights_only=True)

    return page_embeddings, query_embeddings, relevance


def generate_mock_eos_attention(page_embeddings: torch.Tensor) -> list:
    """
    Generate mock EOS attention scores.

    Since we don't have actual EOS attention, use a heuristic:
    - Compute token variance as proxy for information content
    - Higher variance = higher attention score
    """
    num_pages = page_embeddings.shape[0]
    attentions = []

    for i in range(num_pages):
        page = page_embeddings[i]  # [max_tokens, dim]

        # Filter padding
        valid_mask = page.norm(dim=-1) > 1e-6
        valid_tokens = page[valid_mask]

        # Compute variance per token as information proxy
        token_variance = valid_tokens.float().var(dim=-1).cpu().numpy()

        # Normalize to [0, 1]
        if token_variance.max() > 0:
            attn = token_variance / token_variance.max()
        else:
            attn = np.ones_like(token_variance)

        attentions.append(attn)

    return attentions


def evaluate_ptm(
    page_embeddings: torch.Tensor,
    eos_attentions: list,
    query_embeddings: torch.Tensor,
    relevance: torch.Tensor,
    prune_k: float,
    merge_factor: int,
    device: str
) -> dict:
    """Run PTM compression and evaluate."""
    start = time.time()

    # Apply PRUNE-THEN-MERGE
    compressed = prune_then_merge(
        page_embeddings.to(device),
        eos_attentions,
        prune_k=prune_k,
        merge_factor=merge_factor
    )

    # Create query mask
    query_mask = query_embeddings.norm(dim=-1) > 1e-6

    # Compute MaxSim scores
    scores = maxsim_scores(
        query_embeddings,
        query_mask,
        compressed,
        device
    )

    # Compute nDCG@5
    ndcg_5 = ndcg5(scores, relevance)

    elapsed = time.time() - start

    # Estimate memory
    doc_mask = compressed.norm(dim=-1) > 1e-6
    num_tokens_per_page = doc_mask.sum(dim=1).float().mean().item()
    memory_mb = (compressed.shape[0] * num_tokens_per_page * compressed.shape[2] * 4) / (1024**2)

    return {
        "ndcg_at_5": float(ndcg_5),
        "time_sec": elapsed,
        "memory_MB": memory_mb,
        "avg_tokens_per_page": num_tokens_per_page
    }


def main():
    device = "npu:0" if torch.npu.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load physics dataset metadata
    dataset_name = "physics"
    print(f"Loading {dataset_name} dataset metadata...")
    corpus, queries, _, version = load_dataset(dataset_name)
    print(f"  corpus={len(corpus)}  queries={len(queries)}  version={version}\n")

    # Load cached embeddings and relevance
    print("Loading cached embeddings...")
    page_embeddings, query_embeddings, relevance = load_cached_embeddings(dataset_name)
    print(f"  page_embeddings: {tuple(page_embeddings.shape)}")
    print(f"  query_embeddings: {tuple(query_embeddings.shape)}")
    print(f"  relevance: {tuple(relevance.shape)}\n")

    # Generate mock EOS attention
    print("Generating mock EOS attention scores...")
    eos_attentions = generate_mock_eos_attention(page_embeddings)
    print(f"  Generated {len(eos_attentions)} attention arrays\n")

    # Load exact baseline from previous results
    results_dir = Path("tree_index_exp/bench_all") / dataset_name
    exact_result = {"ndcg_at_5": 0.7435}
    tree_results = {}

    if (results_dir / "headtohead.json").exists():
        with open(results_dir / "headtohead.json") as f:
            prev_results = json.load(f)
            if "exact" in prev_results:
                exact_result = prev_results["exact"]
            for method in ["tree", "local_merge", "global_cluster"]:
                if method in prev_results:
                    tree_results[method] = prev_results[method]

    print(f"{'='*70}")
    print(f"  Exact baseline: nDCG@5={exact_result['ndcg_at_5']:.4f}")
    print(f"{'='*70}\n")

    # Run PTM experiments with optimal configs from grid search
    ptm_configs = [
        # (prune_k, merge_factor, description)
        (-0.5, 16, "optimal: k=-0.5, m=16 (target ~32 tokens)"),
        (-1.5, 8, "k=-1.5, m=8 (target ~85 tokens)"),
        (-1.0, 12, "k=-1.0, m=12 (target ~53 tokens)"),
        (0.0, 16, "k=0.0, m=16 (target ~28 tokens)"),
        (0.5, 8, "k=0.5, m=8 (target ~30 tokens)"),
    ]

    ptm_results = {}

    for prune_k, merge_factor, desc in ptm_configs:
        config_name = f"ptm(k={prune_k},m={merge_factor})"
        print(f"[{config_name}] {desc}")

        result = evaluate_ptm(
            page_embeddings,
            eos_attentions,
            query_embeddings,
            relevance,
            prune_k=prune_k,
            merge_factor=merge_factor,
            device=device
        )

        ptm_results[config_name] = result

        retention_pct = 100 * result["ndcg_at_5"] / exact_result["ndcg_at_5"]
        print(f"  nDCG@5={result['ndcg_at_5']:.4f}  "
              f"retention={retention_pct:.1f}%  "
              f"avg_tokens={result['avg_tokens_per_page']:.1f}  "
              f"time={result['time_sec']:.1f}s\n")

    # Print comparison table
    print(f"{'='*70}")
    print(f"{'Method':<25} {'nDCG@5':>10} {'Retention':>12} {'Tokens/Page':>15}")
    print(f"{'='*70}")

    print(f"{'exact':<25} {exact_result['ndcg_at_5']:>10.4f} {'100.0%':>12} {'~770':>15}")

    # Tree results
    for budget in [8, 32, 128]:
        if "tree" in tree_results and str(budget) in tree_results["tree"]:
            ndcg = tree_results["tree"][str(budget)]["ndcg_at_5"]
            retention = 100 * ndcg / exact_result["ndcg_at_5"]
            print(f"{'tree@'+str(budget):<25} {ndcg:>10.4f} {retention:>11.1f}% {budget:>15}")

    # PTM results
    for config_name, result in ptm_results.items():
        ndcg = result["ndcg_at_5"]
        retention = 100 * ndcg / exact_result["ndcg_at_5"]
        tokens = result["avg_tokens_per_page"]
        print(f"{config_name:<25} {ndcg:>10.4f} {retention:>11.1f}% {tokens:>14.1f}")

    print(f"{'='*70}\n")

    # Save results
    output_file = Path("prune_merge") / "ptm_physics_results.json"
    output_data = {
        "dataset": dataset_name,
        "version": version,
        "note": "EOS attention simulated via token variance (not ground truth)",
        "exact": exact_result,
        "ptm": ptm_results,
        "tree_baseline": tree_results
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
