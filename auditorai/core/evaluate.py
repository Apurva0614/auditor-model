"""
Evaluation utilities: reports, plots, and full evaluation runner.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from auditorai.utils.logging import get_logger

logger = get_logger(__name__)


def full_report(
    system: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    human_accuracy: float = 0.72,
) -> dict:
    """Print a formatted evaluation report and return the metrics dict.

    Args:
        system: A trained AuditorSystem instance.
        X_test: Test feature array of shape (n_samples, n_features).
        y_test: Test label array of shape (n_samples,).
        human_accuracy: Simulated human accuracy on suppressed cases.
            Defaults to 0.72.

    Returns:
        A dict of evaluation metrics from system.evaluate().
    """
    metrics = system.evaluate(X_test, y_test, human_accuracy=human_accuracy)

    ai_only = metrics["ai_only_accuracy"] * 100
    joint = metrics["joint_accuracy"] * 100
    gain = metrics["accuracy_gain"] * 100
    auroc = metrics["auditor_auroc"]
    sup_rate = metrics["suppression_rate"] * 100
    n_shown = metrics["n_shown"]
    n_sup = metrics["n_suppressed"]
    prec = metrics["auditor_precision"] * 100
    rec = metrics["auditor_recall"] * 100

    sep = "=" * 50
    arrow = "^" if gain >= 0 else "v"

    print(sep)
    print("  AUDITOR SYSTEM - EVALUATION REPORT")
    print(sep)
    print(f"  AI-only accuracy:         {ai_only:.1f}%")
    print(f"  Joint system accuracy:    {joint:.1f}%  ({arrow}{abs(gain):.1f}%)")
    print(f"  Auditor AUROC:            {auroc:.3f}")
    print(f"  Suppression rate:         {sup_rate:.1f}%")
    print(f"  Cases shown:              {n_shown}")
    print(f"  Cases suppressed:         {n_sup}")
    print(f"  Auditor precision:        {prec:.1f}%")
    print(f"  Auditor recall:           {rec:.1f}%")
    print(sep)

    return metrics


def plot_score_distribution(
    system: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = "outputs/score_dist.png",
) -> None:
    """Plot overlapping histograms of P(wrong) split by actual AI error status.

    Samples where the AI is correct are shown in one colour; samples where
    the AI is wrong are shown in another. A vertical dashed line marks the
    current suppression threshold.

    Args:
        system: A trained AuditorSystem instance.
        X_test: Test feature array of shape (n_samples, n_features).
        y_test: Test label array of shape (n_samples,).
        save_path: Path where the figure will be saved. Defaults to
            "outputs/score_dist.png".
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    result = system.router_.route(X_test)
    scores = result["p_wrong"]
    ai_predictions = result["ai_predictions"]
    is_wrong = ai_predictions != y_test
    threshold = system.router_.threshold

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        scores[~is_wrong],
        bins=30,
        alpha=0.6,
        color="#1D9E75",
        label="AI correct",
        density=True,
    )
    ax.hist(
        scores[is_wrong],
        bins=30,
        alpha=0.6,
        color="#D85A30",
        label="AI wrong",
        density=True,
    )
    ax.axvline(
        threshold,
        color="#534AB7",
        linestyle="--",
        linewidth=2,
        label=f"Threshold tau={threshold:.2f}",
    )
    ax.set_xlabel("P(primary wrong)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Auditor Score Distribution by AI Error Status", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Score distribution plot saved to %s", save_path)


def plot_threshold_sweep(
    system: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    human_accuracy: float = 0.72,
    save_path: str = "outputs/threshold_sweep.png",
) -> None:
    """Plot accuracy gain (%) vs. suppression threshold.

    A horizontal dashed line marks zero gain and a vertical dashed line
    marks the current threshold.

    Args:
        system: A trained AuditorSystem instance.
        X_test: Test feature array of shape (n_samples, n_features).
        y_test: Test label array of shape (n_samples,).
        human_accuracy: Simulated human accuracy. Defaults to 0.72.
        save_path: Path where the figure will be saved. Defaults to
            "outputs/threshold_sweep.png".
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    df = system.router_.sweep_thresholds(X_test, y_test, human_accuracy=human_accuracy)
    threshold = system.router_.threshold

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["tau"], df["accuracy_gain"] * 100, marker="o", color="#534AB7", linewidth=2)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.axvline(threshold, color="#D85A30", linestyle="--", linewidth=2,
               label=f"Current tau={threshold:.2f}")
    ax.set_xlabel("Suppression threshold (tau)", fontsize=12)
    ax.set_ylabel("Accuracy gain (%)", fontsize=12)
    ax.set_title("Joint Accuracy Gain vs. Suppression Threshold", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Threshold sweep plot saved to %s", save_path)


def plot_decision_breakdown(
    system: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str = "outputs/breakdown.png",
) -> None:
    """Plot a bar chart of decision outcomes (shown/suppressed x correct/error).

    The four bars are:
        1. Shown + Correct
        2. Shown + Error
        3. Suppressed + Correct
        4. Suppressed + Error

    Args:
        system: A trained AuditorSystem instance.
        X_test: Test feature array of shape (n_samples, n_features).
        y_test: Test label array of shape (n_samples,).
        save_path: Path where the figure will be saved. Defaults to
            "outputs/breakdown.png".
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    result = system.router_.route(X_test)
    ai_preds = result["ai_predictions"]
    suppress_mask = result["suppress_mask"]
    show_mask = result["show_mask"]

    correct = ai_preds == y_test

    shown_correct = int(np.sum(show_mask & correct))
    shown_error = int(np.sum(show_mask & ~correct))
    suppressed_correct = int(np.sum(suppress_mask & correct))
    suppressed_error = int(np.sum(suppress_mask & ~correct))

    counts = [shown_correct, shown_error, suppressed_correct, suppressed_error]
    labels = ["Shown\n+ Correct", "Shown\n+ Error", "Suppressed\n+ Correct", "Suppressed\n+ Error"]
    colors = ["#1D9E75", "#D85A30", "#7F77DD", "#EF9F27"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=colors, width=0.55)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Decision Breakdown: Shown vs. Suppressed x Correct vs. Error", fontsize=12)
    ax.set_ylim(0, max(counts) * 1.15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Decision breakdown plot saved to %s", save_path)


def run_full_evaluation(
    system: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    human_accuracy: float = 0.72,
    output_dir: str = "outputs",
) -> dict:
    """Run the full evaluation pipeline: report + all three plots.

    Args:
        system: A trained AuditorSystem instance.
        X_test: Test feature array of shape (n_samples, n_features).
        y_test: Test label array of shape (n_samples,).
        human_accuracy: Simulated human accuracy. Defaults to 0.72.
        output_dir: Directory for output plots. Defaults to "outputs".

    Returns:
        Metrics dict from full_report().
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = full_report(system, X_test, y_test, human_accuracy=human_accuracy)
    plot_score_distribution(
        system, X_test, y_test,
        save_path=os.path.join(output_dir, "score_dist.png"),
    )
    plot_threshold_sweep(
        system, X_test, y_test,
        human_accuracy=human_accuracy,
        save_path=os.path.join(output_dir, "threshold_sweep.png"),
    )
    plot_decision_breakdown(
        system, X_test, y_test,
        save_path=os.path.join(output_dir, "breakdown.png"),
    )
    return metrics
