"""Download the minimal public train/evaluation datasets into data_dir/."""

import argparse

from datasets import load_dataset


DATASETS = {
    "train": "vidore/colpali_train_set",
    "eval": "vidore/docvqa_test_subsampled",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--output-dir", default="data_dir")
    args = parser.parse_args()

    names = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for alias, name in names.items():
        dataset = load_dataset(name)
        destination = f"{args.output_dir}/{alias}"
        dataset.save_to_disk(destination)
        print(f"Saved {name} to {destination}")


if __name__ == "__main__":
    main()
