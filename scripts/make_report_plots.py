from __future__ import annotations

import argparse
from pathlib import Path

from src.common.plotting import plot_training_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report plots from training CSV logs")
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--output-root", type=Path, default=Path("results/plots"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for algo in ["dqn", "reinforce"]:
        for run_dir in sorted((args.log_root / algo).glob("*")):
            csv_file = run_dir / "metrics.csv"
            if not csv_file.exists():
                continue
            run_name = run_dir.name
            out_dir = args.output_root / algo / run_name
            plot_training_curve(csv_file, "total_reward", out_dir / "reward.png", f"{run_name} reward")
            plot_training_curve(
                csv_file,
                "moving_avg_reward",
                out_dir / "moving_avg_reward.png",
                f"{run_name} moving average reward",
            )
            if algo == "dqn":
                plot_training_curve(csv_file, "loss", out_dir / "loss.png", f"{run_name} loss")
                plot_training_curve(csv_file, "epsilon", out_dir / "epsilon.png", f"{run_name} epsilon")
            else:
                plot_training_curve(csv_file, "policy_loss", out_dir / "policy_loss.png", f"{run_name} policy loss")
    print(f"Plots saved to {args.output_root}")


if __name__ == "__main__":
    main()
