"""
main.py — CLI entry point for the auditor-model pipeline.

Usage:
    python main.py [--data PATH] [--model TYPE] [--threshold FLOAT]
                   [--human-acc FLOAT] [--no-tune] [--save-dir PATH]
                   [--output-dir PATH]
"""

import argparse
import sys

import numpy as np

from src.evaluate import run_full_evaluation
from src.system import AuditorSystem
from src.utils import get_logger, load_dataset, set_seed, split_data

logger = get_logger("main")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the auditor-model pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=(
            "Path to CSV dataset (last column = label). "
            "If omitted, a synthetic dataset is generated."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "gradient_boosting", "logistic"],
        help="Primary model type.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Starting auditor suppression threshold.",
    )
    parser.add_argument(
        "--human-acc",
        type=float,
        default=0.72,
        dest="human_acc",
        help="Simulated human accuracy on suppressed cases.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip auto-tuning of the suppression threshold.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/models",
        dest="save_dir",
        help="Directory to save trained model files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        dest="output_dir",
        help="Directory to save evaluation plots.",
    )
    return parser


def main() -> None:
    """Main entry point for the auditor-model pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    # Step a: set seed
    set_seed(42)
    logger.info("Random seed set to 42.")

    # Step b: load or generate data
    if args.data is not None:
        logger.info("Loading dataset from %s ...", args.data)
        try:
            X, y = load_dataset(args.data)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        logger.info("Loaded dataset: %d samples, %d features.", X.shape[0], X.shape[1])
    else:
        logger.info("No --data path given. Generating synthetic dataset...")
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=10,
            flip_y=0.08,
            random_state=42,
        )
        y = y.astype(np.float64)
        logger.info("Synthetic dataset: %d samples, %d features.", X.shape[0], X.shape[1])

    # Step c: split data
    logger.info("Splitting data into train / val / test ...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # Step d: create system
    logger.info(
        "Creating AuditorSystem (model=%s, threshold=%.2f) ...",
        args.model,
        args.threshold,
    )
    system = AuditorSystem(
        primary_model_type=args.model,
        auditor_threshold=args.threshold,
    )

    # Step e: train
    logger.info("Training system ...")
    system.train(X_train, y_train, X_val, y_val)

    # Step f: auto-tune
    if not args.no_tune:
        logger.info("Auto-tuning suppression threshold ...")
        best_tau = system.auto_tune(X_val, y_val, human_accuracy=args.human_acc)
        print(f"\nAuto-tuned threshold: tau = {best_tau:.4f}")
    else:
        logger.info("Skipping auto-tune (--no-tune flag set).")

    # Step g: save models
    logger.info("Saving models to %s ...", args.save_dir)
    system.save(args.save_dir)

    # Step h: full evaluation
    logger.info("Running evaluation on test set ...")
    run_full_evaluation(
        system,
        X_test,
        y_test,
        human_accuracy=args.human_acc,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
