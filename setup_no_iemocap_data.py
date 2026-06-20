import argparse
import os

from filter_datasets import filter_datasets
from prepare_english_data import build_english_dataset


def main():
    parser = argparse.ArgumentParser(description="Build English VA data and the no-IEMOCAP filtered split.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data_no_iemocap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--external-dir", default="data/external_english")
    parser.add_argument("--skip-gdrive-download", action="store_true")
    args = parser.parse_args()

    build_english_dataset(
        output_dir=args.data_dir,
        seed=args.seed,
        force=args.force,
        external_dir=args.external_dir,
        skip_gdrive_download=args.skip_gdrive_download,
    )

    fold1 = os.path.join(args.output_dir, "full_dataset_fold1.csv")
    fold2 = os.path.join(args.output_dir, "full_dataset_fold2.csv")
    if not args.force and os.path.isfile(fold1) and os.path.isfile(fold2):
        print(f"no-IEMOCAP data already exists: {args.output_dir}")
        return

    filter_datasets(
        input_dir=args.data_dir,
        output_dir=args.output_dir,
        exclude_patterns=["IEMOCAP"],
        dry_run=False,
    )


if __name__ == "__main__":
    main()
