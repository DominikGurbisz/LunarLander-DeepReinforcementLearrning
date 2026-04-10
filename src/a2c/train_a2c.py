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
from src.a2c.model import PolicyNetwork, ValueNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train A2C on LunarLander-v3")
    parser.add_argument("--exp-name", type=str, default="a2c_default")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=5e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = get_device(args.device)

    env = gym.make("LunarLander-v3")
    env.action_space.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim, args.hidden_size).to(device)
    value_net = ValueNetwork(state_dim, args.hidden_size).to(device)

    policy_optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=args.value_lr)

    run_dir = ensure_dir(Path("logs/a2c") / args.exp_name)
    model_dir = ensure_dir(Path("models/a2c"))

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
        "value_loss",
        "entropy",
        "adv_mean",
        "grad_norm",
    ]

    rewards_history: list[float] = []
    best_reward = -float("inf")

    with CSVLogger(run_dir / "metrics.csv", fieldnames) as logger:
        for episode in range(1, args.episodes + 1):
            state, _ = env.reset(seed=args.seed + episode)

            total_reward = 0.0
            step_count = 0

            episode_total_losses: list[float] = []
            episode_policy_losses: list[float] = []
            episode_value_losses: list[float] = []
            episode_entropies: list[float] = []
            episode_advantages: list[float] = []
            episode_grad_norms: list[float] = []

            for step in range(args.max_steps):
                step_count = step + 1

                st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                logits = policy(st)
                value = value_net(st).squeeze(1)

                dist = Categorical(logits=logits)
                action = dist.sample()

                next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
                done = terminated or truncated
                total_reward += float(reward)

                with torch.no_grad():
                    if done:
                        next_value = torch.zeros(1, dtype=torch.float32, device=device)
                    else:
                        next_st = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                        next_value = value_net(next_st).squeeze(1)

                reward_tensor = torch.tensor([float(reward)], dtype=torch.float32, device=device)
                done_tensor = torch.tensor([1.0 if done else 0.0], dtype=torch.float32, device=device)

                td_target = reward_tensor + args.gamma * next_value * (1.0 - done_tensor)
                advantage = td_target - value

                policy_loss = -(dist.log_prob(action) * advantage.detach()).mean()
                value_loss = F.mse_loss(value, td_target)
                entropy = dist.entropy().mean()

                total_loss = (
                    policy_loss
                    + args.value_loss_coef * value_loss
                    - args.entropy_coef * entropy
                )

                policy_optimizer.zero_grad()
                value_optimizer.zero_grad()
                total_loss.backward()

                policy_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                value_grad_norm = torch.nn.utils.clip_grad_norm_(value_net.parameters(), 10.0)
                grad_norm = float(max(float(policy_grad_norm), float(value_grad_norm)))

                policy_optimizer.step()
                value_optimizer.step()

                episode_total_losses.append(float(total_loss.item()))
                episode_policy_losses.append(float(policy_loss.item()))
                episode_value_losses.append(float(value_loss.item()))
                episode_entropies.append(float(entropy.item()))
                episode_advantages.append(float(advantage.mean().item()))
                episode_grad_norms.append(grad_norm)

                state = next_state
                if done:
                    break

            rewards_history.append(total_reward)
            ma = moving_average(rewards_history, window=50)[-1]

            logger.log(
                {
                    "episode": episode,
                    "total_reward": total_reward,
                    "moving_avg_reward": ma,
                    "episode_length": step_count,
                    "loss": sum(episode_total_losses) / max(1, len(episode_total_losses)),
                    "learning_rate": policy_optimizer.param_groups[0]["lr"],
                    "seed": args.seed,
                    "exp_name": args.exp_name,
                    "policy_loss": sum(episode_policy_losses) / max(1, len(episode_policy_losses)),
                    "value_loss": sum(episode_value_losses) / max(1, len(episode_value_losses)),
                    "entropy": sum(episode_entropies) / max(1, len(episode_entropies)),
                    "adv_mean": sum(episode_advantages) / max(1, len(episode_advantages)),
                    "grad_norm": sum(episode_grad_norms) / max(1, len(episode_grad_norms)),
                }
            )

            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(policy.state_dict(), model_dir / f"{args.exp_name}_best.pt")
                torch.save(value_net.state_dict(), model_dir / f"{args.exp_name}_value_best.pt")

            if episode % 25 == 0:
                print(f"[A2C] ep={episode:4d} reward={total_reward:8.2f} ma50={ma:8.2f}")

    final_policy_path = model_dir / f"{args.exp_name}.pt"
    final_value_path = model_dir / f"{args.exp_name}_value.pt"

    torch.save(policy.state_dict(), final_policy_path)
    torch.save(value_net.state_dict(), final_value_path)

    save_json(
        run_dir / "summary.json",
        {
            "algorithm": "a2c",
            "exp_name": args.exp_name,
            "seed": args.seed,
            "final_model": str(final_policy_path),
            "final_value_model": str(final_value_path),
            "episodes": args.episodes,
            "best_episode_reward": best_reward,
            "final_moving_avg_50": moving_average(rewards_history, 50)[-1],
            "hyperparameters": vars(args),
        },
    )

    env.close()
    print(f"Saved A2C policy to {final_policy_path}")


if __name__ == "__main__":
    main()