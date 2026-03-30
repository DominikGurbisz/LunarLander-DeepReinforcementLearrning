from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one demo video for each trained config")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-size-dqn", type=int, default=128)
    parser.add_argument("--hidden-size-reinforce", type=int, default=128)
    return parser.parse_args()


def run_demo(algo: str, model_path: Path, folder: Path, prefix: str, seed: int, hidden_size: int) -> None:
    if not model_path.exists():
        print(f"Missing model, skip: {model_path}")
        return
    cmd = [
        sys.executable,
        "play_demo.py",
        "--algo",
        algo,
        "--model-path",
        str(model_path),
        "--seed",
        str(seed),
        "--hidden-size",
        str(hidden_size),
        "--record-video",
        "--video-folder",
        str(folder),
        "--video-prefix",
        prefix,
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    dqn_labels = ["epsilon_decay_low", "epsilon_decay_opt", "epsilon_decay_high", "learning_rate_low", "learning_rate_opt", "learning_rate_high"]
    reinf_labels = ["gamma_low", "gamma_opt", "gamma_high", "hidden_size_low", "hidden_size_opt", "hidden_size_high"]

    for label in dqn_labels:
        exp = f"dqn_{label}_seed{args.seed}"
        run_demo(
            "dqn",
            Path("models/dqn") / f"{exp}.pt",
            Path("videos/dqn") / label,
            f"dqn_{label}",
            args.seed,
            args.hidden_size_dqn,
        )

    for label in reinf_labels:
        exp = f"reinforce_{label}_seed{args.seed}"
        hidden_size = args.hidden_size_reinforce
        if label == "hidden_size_low":
            hidden_size = 64
        elif label == "hidden_size_high":
            hidden_size = 256
        run_demo(
            "reinforce",
            Path("models/reinforce") / f"{exp}.pt",
            Path("videos/reinforce") / label,
            f"reinforce_{label}",
            args.seed,
            hidden_size,
        )


if __name__ == "__main__":
    main()
