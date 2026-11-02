#!/usr/bin/env python3
"""One-shot download: ColQwen2.5 weights + ViDoRe V3 evaluation datasets.

Usage:
    python experiment/scripts/download.py
    python experiment/scripts/download.py --datasets physics energy
    python experiment/scripts/download.py --cache-dir /mnt/data/.cache/huggingface
    python experiment/scripts/download.py --hf-endpoint https://huggingface.co

Outputs:
    <cache-dir>/hub/models--vidore--colqwen2.5-base/
    <cache-dir>/hub/models--vidore--colqwen2.5-v0.1/
    <cache-dir>/vidore___vidore_v3_<dataset>/
"""
import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent.parent

ALL_DATASETS = [
    "physics", "computer_science", "energy",
    "finance_en", "finance_fr", "hr", "industrial", "pharmaceuticals",
]

# corpus shard counts per dataset
CORPUS_SHARDS = {
    "physics": 3, "energy": 3, "finance_fr": 3,
    "computer_science": 2, "pharmaceuticals": 2,
    "finance_en": 2, "hr": 1, "industrial": 1,
}

MODEL_FILES = [
    "config.json", "tokenizer_config.json", "tokenizer.json",
    "preprocessor_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "model.safetensors.index.json",
]
ADAPTER_FILES = [
    "adapter_config.json", "adapter_model.safetensors",
    "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
]
ADAPTER_SNAPSHOT = "4379ecfd4c9446836ad0ef4dd7390b1c748f68d8"
MODEL_SHARDS = 8


def _curl(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"    [skip] {dest.name}")
        return True
    print(f"    [dl]   {url}")
    r = subprocess.run(
        ["curl", "-sSL", "--retry", "5", "--retry-delay", "3", "-C", "-", "-o", str(dest), url]
    )
    return r.returncode == 0


def download_weights(cache_dir: Path, hf_endpoint: str) -> None:
    base = hf_endpoint.rstrip("/")

    # base model
    snap_dir = cache_dir / "hub/models--vidore--colqwen2.5-base/snapshots/main"
    snap_dir.mkdir(parents=True, exist_ok=True)
    print("--- ColQwen2.5 base weights ---")
    for f in MODEL_FILES:
        _curl(f"{base}/models/vidore/colqwen2.5-base/resolve/main/{f}", snap_dir / f)
    for i in range(1, MODEL_SHARDS + 1):
        fname = f"model-{i:05d}-of-{MODEL_SHARDS:05d}.safetensors"
        _curl(f"{base}/models/vidore/colqwen2.5-base/resolve/main/{fname}", snap_dir / fname)

    # adapter
    adpt_dir = cache_dir / f"hub/models--vidore--colqwen2.5-v0.1/snapshots/{ADAPTER_SNAPSHOT}"
    adpt_dir.mkdir(parents=True, exist_ok=True)
    print("\n--- ColQwen2.5-v0.1 adapter ---")
    for f in ADAPTER_FILES:
        _curl(f"{base}/models/vidore/colqwen2.5-v0.1/resolve/main/{f}", adpt_dir / f)


def _write_arrow(parquet_files: list[Path], dest: Path) -> None:
    tables = [pq.read_table(str(f)) for f in parquet_files]
    tbl = pa.concat_tables(tables)
    with pa.ipc.new_stream(str(dest), tbl.schema) as w:
        w.write_table(tbl)
    print(f"    [conv] {len(tbl)} rows → {dest}")


def download_dataset(name: str, cache_dir: Path, hf_endpoint: str) -> None:
    base = hf_endpoint.rstrip("/")
    repo = f"vidore/vidore_v3_{name}"
    ds_cache = cache_dir / f"vidore___vidore_v3_{name}"
    n_shards = CORPUS_SHARDS.get(name, 2)

    print(f"\n--- Dataset: {name} ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for split in ("corpus", "queries", "qrels"):
            arrow = ds_cache / split / "data.arrow"
            if arrow.exists():
                print(f"  [skip] {split} (cached)")
                continue
            arrow.parent.mkdir(parents=True, exist_ok=True)

            shards = n_shards if split == "corpus" else 1
            total = n_shards if split == "corpus" else 1
            parquets = []
            for i in range(shards):
                fname = f"data-{i:05d}-of-{total:05d}.parquet"
                dest = tmp / split / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                ok = _curl(f"{base}/datasets/{repo}/resolve/main/{split}/{fname}", dest)
                if ok and dest.exists():
                    parquets.append(dest)

            if parquets:
                _write_arrow(parquets, arrow)
            else:
                print(f"  [WARN] no parquet shards downloaded for {name}/{split}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cache-dir", type=Path,
        default=ROOT / ".cache/huggingface",
        help="Root HF cache directory (default: <repo>/.cache/huggingface)",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=ALL_DATASETS, default=ALL_DATASETS,
        metavar="DS",
        help=f"Datasets to download (default: all). Choices: {', '.join(ALL_DATASETS)}",
    )
    parser.add_argument(
        "--hf-endpoint", default="https://hf-mirror.com",
        help="HuggingFace endpoint (default: https://hf-mirror.com)",
    )
    parser.add_argument(
        "--skip-weights", action="store_true",
        help="Skip downloading model weights (datasets only)",
    )
    parser.add_argument(
        "--skip-datasets", action="store_true",
        help="Skip downloading datasets (weights only)",
    )
    args = parser.parse_args()

    print("=== Download: ColQwen2.5 weights + ViDoRe V3 datasets ===")
    print(f"Cache dir   : {args.cache_dir}")
    print(f"HF endpoint : {args.hf_endpoint}")
    print(f"Datasets    : {' '.join(args.datasets)}")
    print()

    if not args.skip_weights:
        download_weights(args.cache_dir, args.hf_endpoint)

    if not args.skip_datasets:
        for ds in args.datasets:
            download_dataset(ds, args.cache_dir, args.hf_endpoint)

    print("\n=== Download complete ===")
    print(f"Model base : {args.cache_dir}/hub/models--vidore--colqwen2.5-base")
    print(f"Adapter    : {args.cache_dir}/hub/models--vidore--colqwen2.5-v0.1")
    print(f"Datasets   : {args.cache_dir}/vidore___vidore_v3_*")


if __name__ == "__main__":
    main()
