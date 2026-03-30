from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch
from torch.distributions import Categorical

from src.common.utils import get_device
from src.common.video import make_video_env
from src.dqn.model import QNetwork
from src.reinforce.model import PolicyNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play one demo episode with a saved agent")
    parser.add_argument("--algo", choices=["dqn", "reinforce"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--render", action="store_true", help="Use human rendering")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-folder", type=Path, default=Path("videos/demo"))
    parser.add_argument("--video-prefix", type=str, default="demo")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    if args.record_video:
        env = make_video_env("LunarLander-v3", args.video_folder, args.video_prefix, args.seed)
    else:
        render_mode = "human" if args.render else None
        env = gym.make("LunarLander-v3", render_mode=render_mode)

    device = get_device(args.device)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if args.algo == "dqn":
        model = QNetwork(state_dim, action_dim, args.hidden_size).to(device)
    else:
        model = PolicyNetwork(state_dim, action_dim, args.hidden_size).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    state, _ = env.reset(seed=args.seed)
    done = False
    total_reward = 0.0

    while not done:
        with torch.no_grad():
            st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            if args.algo == "dqn":
                action = int(torch.argmax(model(st), dim=1).item())
            else:
                dist = Categorical(logits=model(st))
                action = int(torch.argmax(dist.probs, dim=1).item())

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()
    print(f"Demo finished. Algo={args.algo} Reward={total_reward:.2f}")


if __name__ == "__main__":
    main()
