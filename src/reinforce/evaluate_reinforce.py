from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

from src.common.utils import ensure_dir, get_device, save_json
from src.common.video import make_video_env
from src.reinforce.model import PolicyNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained REINFORCE model")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-folder", type=Path, default=Path("videos/reinforce/eval"))
    parser.add_argument("--video-prefix", type=str, default="reinforce_eval")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--results-path", type=Path, default=Path("results/tables/reinforce_eval_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    device = get_device(args.device)
    env = (
        make_video_env("LunarLander-v3", args.video_folder, args.video_prefix, args.seed)
        if args.record_video
        else gym.make("LunarLander-v3")
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = PolicyNetwork(state_dim, action_dim, args.hidden_size).to(device)
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    policy.eval()

    rewards = []
    for ep in range(args.episodes):
        state, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                dist = Categorical(logits=policy(s))
                action = int(torch.argmax(dist.probs, dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)

    env.close()
    summary = {
        "algorithm": "reinforce",
        "model_path": str(args.model_path),
        "episodes": args.episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "seed": args.seed,
    }
    ensure_dir(args.results_path.parent)
    save_json(args.results_path, summary)
    print(summary)


if __name__ == "__main__":
    main()
