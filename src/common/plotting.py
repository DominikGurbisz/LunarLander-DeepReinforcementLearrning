from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.common.utils import ensure_dir


def plot_training_curve(
    csv_path: Path | str,
    metric: str,
    output_path: Path | str,
    title: str,
    group_col: str | None = None,
) -> None:
    df = pd.read_csv(csv_path)
    ensure_dir(Path(output_path).parent)

    plt.figure(figsize=(10, 5))
    if group_col and group_col in df.columns:
        for name, group in df.groupby(group_col):
            plt.plot(group["episode"], group[metric], label=str(name), alpha=0.9)
        plt.legend()
    else:
        plt.plot(df["episode"], df[metric], label=metric)

    plt.xlabel("Episode")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
