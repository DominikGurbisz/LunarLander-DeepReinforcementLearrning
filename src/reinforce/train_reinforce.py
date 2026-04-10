from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from src.common.logger import CSVLogger
from src.common.seed import set_global_seed
from src.common.utils import ensure_dir, get_device, moving_average, save_json
from src.reinforce.model import PolicyNetwork, ValueNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train REINFORCE on LunarLander-v3")
    parser.add_argument("--exp-name", type=str, default="reinforce_default")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--value-lr", type=float, default=5e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--use-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--normalize-returns", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    out = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    out.reverse()
    return out


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = get_device(args.device)

    env = gym.make("LunarLander-v3")
    env.action_space.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim, args.hidden_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    value_net = ValueNetwork(state_dim, args.hidden_size).to(device) if args.use_baseline else None
    value_optimizer = optim.Adam(value_net.parameters(), lr=args.value_lr) if value_net is not None else None

    run_dir = ensure_dir(Path("logs/reinforce") / args.exp_name)
    model_dir = ensure_dir(Path("models/reinforce"))

    fieldnames = [
        "episode",
        "total_reward",
        "moving_avg_reward",
        "episode_length",
        "loss",
        "learning_rate",
        "seed",
        "exp_name",
        "policy_loss",
        "return_mean",
        "return_std",
        "grad_norm",
        "value_loss",
        "adv_mean",
        "adv_std",
    ]

    rewards_history: list[float] = []
    best_reward = -float("inf")

    with CSVLogger(run_dir / "metrics.csv", fieldnames) as logger:
        for episode in range(1, args.episodes + 1):
            state, _ = env.reset(seed=args.seed + episode)
            states: list[torch.Tensor] = []
            log_probs: list[torch.Tensor] = []
            rewards: list[float] = []
            total_reward = 0.0

            for step in range(args.max_steps):
                st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                states.append(st.squeeze(0))
                logits = policy(st)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))

                state, reward, terminated, truncated, _ = env.step(int(action.item()))
                rewards.append(float(reward))
                total_reward += float(reward)
                if terminated or truncated:
                    break

            returns = torch.tensor(discounted_returns(rewards, args.gamma), dtype=torch.float32, device=device)
            if args.normalize_returns and len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            values = None
            value_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            advantages = returns
            if value_net is not None and value_optimizer is not None and states:
                states_tensor = torch.stack(states)
                values = value_net(states_tensor).squeeze(1)
                value_loss = F.mse_loss(values, returns)
                advantages = returns - values.detach()

            policy_loss = -(torch.stack(log_probs) * advantages).sum()
            total_loss = policy_loss + args.value_loss_coef * value_loss

            optimizer.zero_grad()
            if value_optimizer is not None:
                value_optimizer.zero_grad()
            total_loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0).item())
            if value_net is not None:
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), 10.0)
            optimizer.step()
            if value_optimizer is not None:
                value_optimizer.step()

            rewards_history.append(total_reward)
            ma = moving_average(rewards_history, window=50)[-1]
            ret_mean = float(returns.mean().item()) if len(returns) else 0.0
            ret_std = float(returns.std().item()) if len(returns) > 1 else 0.0
            adv_mean = float(advantages.mean().item()) if len(advantages) else 0.0
            adv_std = float(advantages.std().item()) if len(advantages) > 1 else 0.0

            logger.log(
                {
                    "episode": episode,
                    "total_reward": total_reward,
                    "moving_avg_reward": ma,
                    "episode_length": step + 1,
                    "loss": float(policy_loss.item()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "seed": args.seed,
                    "exp_name": args.exp_name,
                    "policy_loss": float(policy_loss.item()),
                    "return_mean": ret_mean,
                    "return_std": ret_std,
                    "grad_norm": grad_norm,
                    "value_loss": float(value_loss.item()),
                    "adv_mean": adv_mean,
                    "adv_std": adv_std,
                }
            )

            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(policy.state_dict(), model_dir / f"{args.exp_name}_best.pt")

            if episode % 25 == 0:
                print(f"[REINFORCE] ep={episode:4d} reward={total_reward:8.2f} ma50={ma:8.2f}")

    final_model_path = model_dir / f"{args.exp_name}.pt"
    torch.save(policy.state_dict(), final_model_path)
    save_json(
        run_dir / "summary.json",
        {
            "algorithm": "reinforce",
            "exp_name": args.exp_name,
            "seed": args.seed,
            "final_model": str(final_model_path),
            "episodes": args.episodes,
            "best_episode_reward": best_reward,
            "final_moving_avg_50": moving_average(rewards_history, 50)[-1],
            "hyperparameters": vars(args),
        },
    )
    env.close()
    print(f"Saved REINFORCE model to {final_model_path}")


if __name__ == "__main__":
    main()
