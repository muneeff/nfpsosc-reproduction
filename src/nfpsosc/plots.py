from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_prediction_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    y = np.asarray(y_true).reshape(-1)
    p = np.asarray(y_pred).reshape(-1)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(y, label="Actual")
    ax.plot(p, label="Predicted")
    ax.set_title(title)
    ax.set_xlabel("Test sample")
    ax.set_ylabel("Revenue")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_residual_plot(
    residuals: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    e = np.asarray(residuals).reshape(-1)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(0.0, linewidth=1)
    ax.plot(e)
    ax.set_title(title)
    ax.set_xlabel("Test sample")
    ax.set_ylabel("Residual")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_pso_history(history: dict[str, np.ndarray], output_dir: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for key in (
        "best_cost",
        "mean_velocity_norm",
        "swarm_diameter",
        "best_parameter_drift",
        "boundary_hits",
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(history[key])
        ax.set_title(key.replace("_", " ").title())
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(directory / f"pso_{key}.png", dpi=180)
        plt.close(fig)
