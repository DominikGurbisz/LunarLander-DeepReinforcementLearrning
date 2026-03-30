from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], dry_run: bool = False) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all 12 core hyperparameter experiments")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes-dqn", type=int, default=600)
    parser.add_argument("--episodes-reinforce", type=int, default=800)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dqn_configs = [
        ("epsilon_decay_low", ["--epsilon-decay", "0.990"]),
        ("epsilon_decay_opt", ["--epsilon-decay", "0.995"]),
        ("epsilon_decay_high", ["--epsilon-decay", "0.999"]),
        ("learning_rate_low", ["--lr", "0.0003"]),
        ("learning_rate_opt", ["--lr", "0.001"]),
        ("learning_rate_high", ["--lr", "0.003"]),
    ]

    reinforce_configs = [
        ("gamma_low", ["--gamma", "0.90"]),
        ("gamma_opt", ["--gamma", "0.99"]),
        ("gamma_high", ["--gamma", "0.999"]),
        ("hidden_size_low", ["--hidden-size", "64"]),
        ("hidden_size_opt", ["--hidden-size", "128"]),
        ("hidden_size_high", ["--hidden-size", "256"]),
    ]

    for name, extra in dqn_configs:
        exp = f"dqn_{name}_seed{args.seed}"
        model_path = Path("models/dqn") / f"{exp}.pt"
        if args.skip_existing and model_path.exists():
            print(f"Skip existing: {model_path}")
            continue
        cmd = [
            sys.executable,
            "-m",
            "src.dqn.train_dqn",
            "--exp-name",
            exp,
            "--seed",
            str(args.seed),
            "--episodes",
            str(args.episodes_dqn),
            *extra,
        ]
        run(cmd, args.dry_run)

    for name, extra in reinforce_configs:
        exp = f"reinforce_{name}_seed{args.seed}"
        model_path = Path("models/reinforce") / f"{exp}.pt"
        if args.skip_existing and model_path.exists():
            print(f"Skip existing: {model_path}")
            continue
        cmd = [
            sys.executable,
            "-m",
            "src.reinforce.train_reinforce",
            "--exp-name",
            exp,
            "--seed",
            str(args.seed),
            "--episodes",
            str(args.episodes_reinforce),
            "--normalize-returns",
            *extra,
        ]
        run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
