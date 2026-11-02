#!/usr/bin/env python3
"""Extract ColQwen2.5 embeddings for ViDoRe V3 datasets.

Usage:
    python experiment/scripts/embedding.py --data-dir /path/to/hf/cache
    python experiment/scripts/embedding.py \\
        --data-dir /mnt/data/.cache/huggingface \\
        --model-base /mnt/data/.cache/huggingface/hub/models--vidore--colqwen2.5-base/snapshots/main \\
        --model-adpt /mnt/data/.cache/huggingface/hub/models--vidore--colqwen2.5-v0.1/snapshots/4379ecfd4c9446836ad0ef4dd7390b1c748f68d8 \\
        --out-dir /mnt/data/embeddings \\
        --npu-phys 9 \\
        --datasets physics energy

Outputs per dataset:
    <out-dir>/<dataset>/page_embeddings.pt   (P, T, 128) float32
    <out-dir>/<dataset>/query_embeddings.pt  (Q, Tq, 128) float32
    <out-dir>/<dataset>/relevance.pt         (Q, P) float32
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

ADAPTER_SNAPSHOT = "4379ecfd4c9446836ad0ef4dd7390b1c748f68d8"

ALL_DATASETS = [
    "physics", "computer_science", "energy",
    "finance_en", "finance_fr", "hr", "industrial", "pharmaceuticals",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / ".cache/huggingface",
        help="Root HF cache dir containing vidore___vidore_v3_* subdirs",
    )
    parser.add_argument(
        "--model-base", type=Path,
        default=ROOT / ".cache/huggingface/hub/models--vidore--colqwen2.5-base/snapshots/main",
        help="Path to colqwen2.5-base snapshot directory",
    )
    parser.add_argument(
        "--model-adpt", type=Path,
        default=ROOT / f".cache/huggingface/hub/models--vidore--colqwen2.5-v0.1/snapshots/{ADAPTER_SNAPSHOT}",
        help="Path to colqwen2.5-v0.1 adapter snapshot directory",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "experiment/embeddings/colqwen2.5_v3",
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["all"], metavar="DS",
        help=f"Dataset names or 'all' (default: all). Choices: {', '.join(ALL_DATASETS)}",
    )
    parser.add_argument(
        "--npu-phys", type=int, default=8,
        help="Physical NPU card index for ASCEND_RT_VISIBLE_DEVICES (default: 8)",
    )
    args = parser.parse_args()

    datasets = ALL_DATASETS if args.datasets == ["all"] else args.datasets
    for ds in datasets:
        if ds not in ALL_DATASETS:
            parser.error(f"Unknown dataset: {ds}. Choices: {', '.join(ALL_DATASETS)}")

    if not args.data_dir.is_dir():
        parser.error(f"--data-dir not found: {args.data_dir}")
    if not args.model_base.is_dir():
        parser.error(f"--model-base not found: {args.model_base}")
    if not args.model_adpt.is_dir():
        parser.error(f"--model-adpt not found: {args.model_adpt}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== ColQwen2.5 Embedding Extraction ===")
    print(f"Data dir   : {args.data_dir}")
    print(f"Model base : {args.model_base}")
    print(f"Adapter    : {args.model_adpt}")
    print(f"Output     : {args.out_dir}")
    print(f"Datasets   : {' '.join(datasets)}")
    print(f"NPU card   : {args.npu_phys}  (logical npu:0 inside process)")
    print()

    # Set NPU visibility before importing torch_npu
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu_phys)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Import encode module and call main() with explicit path args,
    # bypassing any environment-variable or compiled-default path logic.
    sys.path.insert(0, str(ROOT))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "encode_colqwen25_v3",
        ROOT / "experiment/scripts/encode_colqwen25_v3.py",
    )
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)

    ds_arg = "all" if datasets == ALL_DATASETS else ",".join(datasets)
    enc.main([
        "--datasets",   ds_arg,
        "--model-base", str(args.model_base),
        "--model-adpt", str(args.model_adpt),
        "--data-dir",   str(args.data_dir),
        "--out-dir",    str(args.out_dir),
    ])

    print("\n=== Embedding extraction complete ===")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
