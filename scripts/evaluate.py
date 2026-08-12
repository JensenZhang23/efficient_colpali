"""Evaluate a checkpoint on a ViDoRe-format query/image dataset."""

import argparse
import json
import math
from pathlib import Path

from datasets import load_dataset, load_from_disk

from colpali_engine.workflow import encode, load_model, score_queries


def retrieval_metrics(scores, document_ids, relevant_ids, ks=(1, 5, 10)):
    metrics = {}
    rankings = scores.argsort(dim=1, descending=True)
    ranks = []
    for query_index, relevant_id in enumerate(relevant_ids):
        rank = next(
            index + 1
            for index, doc_index in enumerate(rankings[query_index].tolist())
            if document_ids[doc_index] == relevant_id
        )
        ranks.append(rank)

    for k in ks:
        metrics[f"recall_at_{k}"] = sum(rank <= k for rank in ranks) / len(ranks)
        metrics[f"mrr_at_{k}"] = sum(1 / rank if rank <= k else 0 for rank in ranks) / len(ranks)
        metrics[f"ndcg_at_{k}"] = sum(1 / math.log2(rank + 1) if rank <= k else 0 for rank in ranks) / len(ranks)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["colpali", "colqwen2"], default="colqwen2")
    parser.add_argument("--model-name", help="Hugging Face model ID or local checkpoint")
    parser.add_argument("--dataset", default="vidore/docvqa_test_subsampled")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    path = Path(args.dataset)
    dataset = load_from_disk(str(path)) if path.exists() else load_dataset(args.dataset, split=args.split)
    if hasattr(dataset, "keys") and args.split in dataset:
        dataset = dataset[args.split]
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # Encode each physical page once while retaining every query and its relevant page ID.
    id_column = "image_filename" if "image_filename" in dataset.column_names else "docId"
    unique = {}
    for row in dataset:
        unique.setdefault(str(row[id_column]), row["image"])

    model, processor, device = load_model(args.model_type, args.model_name, args.device)
    documents = encode(unique.values(), model, processor, device, "images", args.batch_size)
    queries = encode(dataset["query"], model, processor, device, "queries", args.batch_size)
    scores = score_queries(processor, queries, documents, device)
    metrics = retrieval_metrics(scores, list(unique), [str(value) for value in dataset[id_column]])

    Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
