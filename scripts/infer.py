"""Encode local images and rank them for one text query."""

import argparse
from pathlib import Path

from PIL import Image

from colpali_engine.workflow import encode, load_model, score_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["colpali", "colqwen2"], default="colqwen2")
    parser.add_argument("--model-name", help="Hugging Face model ID or local checkpoint")
    parser.add_argument("--query", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    paths = [Path(path) for path in args.images]
    images = [Image.open(path).convert("RGB") for path in paths]
    model, processor, device = load_model(args.model_type, args.model_name, args.device)
    documents = encode(images, model, processor, device, "images", args.batch_size)
    queries = encode([args.query], model, processor, device, "queries", args.batch_size)
    scores = score_queries(processor, queries, documents, device)[0]

    for index in scores.argsort(descending=True).tolist():
        print(f"{scores[index].item():.6f}\t{paths[index]}")


if __name__ == "__main__":
    main()
