"""
Grid search for optimal PRUNE-THEN-MERGE parameters.

Goal: Find (prune_k, merge_factor) that matches tree@32 performance
      while using similar token budget (~32 tokens/page).
"""
import json
import time
from pathlib import Path

import torch
import numpy as np

from prune_merge.run_ptm_physics import (
    load_cached_embeddings,
    generate_mock_eos_attention,
    evaluate_ptm
)


def grid_search():
    """Systematic grid search over PTM parameters."""
    device = "npu:0"
    dataset_name = "physics"

    print("Loading cached embeddings...")
    page_embeddings, query_embeddings, relevance = load_cached_embeddings(dataset_name)

    # Sample for faster iteration
    sample_pages = 50
    sample_queries = 100
    page_embeddings = page_embeddings[:sample_pages]
    query_embeddings = query_embeddings[:sample_queries]
    relevance = relevance[:sample_queries, :sample_pages]

    print(f"Sampled: {sample_pages} pages, {sample_queries} queries")
    print(f"Generating mock EOS attention...")
    eos_attentions = generate_mock_eos_attention(page_embeddings)

    # Load exact baseline
    results_dir = Path("tree_index_exp/bench_all") / dataset_name
    with open(results_dir / "headtohead.json") as f:
        prev_results = json.load(f)
        exact_ndcg = prev_results["exact"]["ndcg_at_5"]
        tree32_ndcg = prev_results["tree"]["32"]["ndcg_at_5"]

    print(f"\nBaselines (full dataset):")
    print(f"  exact: nDCG@5={exact_ndcg:.4f}")
    print(f"  tree@32: nDCG@5={tree32_ndcg:.4f} (95.7% retention)")
    print(f"\nTarget: ~32 tokens/page, >90% retention\n")

    # Grid search parameters
    # prune_k: negative values = keep more tokens
    # -2.0 → keep ~98%, -1.0 → keep ~84%, 0.0 → keep ~50%, 1.0 → keep ~16%
    prune_k_values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5]
    merge_factor_values = [2, 4, 6, 8, 12, 16]

    results = []
    best_config = None
    best_score = 0.0

    print(f"{'prune_k':<10} {'merge_m':<10} {'tokens/pg':<12} {'nDCG@5':<10} {'retain%':<10}")
    print("=" * 70)

    for prune_k in prune_k_values:
        for merge_m in merge_factor_values:
            result = evaluate_ptm(
                page_embeddings,
                eos_attentions,
                query_embeddings,
                relevance,
                prune_k=prune_k,
                merge_factor=merge_m,
                device=device
            )

            ndcg = result["ndcg_at_5"]
            tokens = result["avg_tokens_per_page"]
            retention = 100 * ndcg / exact_ndcg

            results.append({
                "prune_k": prune_k,
                "merge_factor": merge_m,
                "ndcg_at_5": ndcg,
                "tokens_per_page": tokens,
                "retention_pct": retention,
                "memory_MB": result["memory_MB"],
                "time_sec": result["time_sec"]
            })

            print(f"{prune_k:<10.1f} {merge_m:<10} {tokens:<12.1f} {ndcg:<10.4f} {retention:<10.1f}")

            # Track best config (closest to tree@32 with similar token budget)
            if 25 <= tokens <= 40 and ndcg > best_score:
                best_score = ndcg
                best_config = (prune_k, merge_m, tokens, ndcg)

    print("=" * 70)

    if best_config:
        pk, mm, tk, nd = best_config
        print(f"\n🏆 Best config: prune_k={pk}, merge_m={mm}")
        print(f"   tokens/page={tk:.1f}, nDCG@5={nd:.4f} ({100*nd/exact_ndcg:.1f}% retention)")
        print(f"   vs tree@32: {tk:.1f} tokens, {nd:.4f} nDCG (tree: 32 tokens, {tree32_ndcg:.4f} nDCG)")

    # Save results
    output_file = Path("prune_merge") / "grid_search_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "sample_pages": sample_pages,
            "sample_queries": sample_queries,
            "baselines": {
                "exact": exact_ndcg,
                "tree@32": tree32_ndcg
            },
            "grid_results": results,
            "best_config": best_config if best_config else None
        }, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    grid_search()
