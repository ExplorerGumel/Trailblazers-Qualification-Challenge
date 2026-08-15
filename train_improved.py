"""Command-line entrypoint for the reliable tree-based regression workflow."""

from __future__ import annotations

import argparse

from src.tree_pipeline import run_tree_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate tabular regressors")
    parser.add_argument("--train", required=True, help="Path to the training CSV")
    parser.add_argument("--target-col", default="target", help="Name of the target column")
    parser.add_argument("--output-dir", default="models", help="Directory for model artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-tune", action="store_true", help="Skip the randomized HGB search"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> dict:
    return run_tree_experiment(
        train_path=args.train,
        target_col=args.target_col,
        output_dir=args.output_dir,
        random_state=args.random_state,
        tune=not args.no_tune,
    )


if __name__ == "__main__":
    result = main(parse_args())
    print("RESULTS:", result)
