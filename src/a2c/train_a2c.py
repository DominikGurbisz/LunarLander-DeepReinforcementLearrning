from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from src.a2c.model import PolicyNetwork, ValueNetwork
from src.common.logger import CSVLogger
from src.common.seed import set_global_seed
from src.common.utils import ensure_dir, get_device, moving_average, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train A2C on LunarLander-v3")
    parser.add_argument("--exp-name", type=str, default="a2c_default")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=800_000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=5e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=1e-2)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def make_env() -> gym.Env:
    return gym.make("LunarLander-v3")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = get_device(args.device)

    envs = gym.vector.SyncVectorEnv([make_env for _ in range(args.num_envs)])
    envs.action_space.seed(args.seed)

    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n

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
    episode_count = 0

    next_obs, _ = envs.reset(seed=args.seed)
    global_step = 0

    ep_rewards = np.zeros(args.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int32)

    last_finished_reward = 0.0
    last_finished_length = 0

    with CSVLogger(run_dir / "metrics.csv", fieldnames) as logger:
        while global_step < args.total_timesteps:
            obs_buf: list[torch.Tensor] = []
            actions_buf: list[torch.Tensor] = []
            logprob_buf: list[torch.Tensor] = []
            rewards_buf: list[torch.Tensor] = []
            dones_buf: list[torch.Tensor] = []
            values_buf: list[torch.Tensor] = []
            entropy_buf: list[torch.Tensor] = []

            for _ in range(args.n_steps):
                global_step += args.num_envs
                obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

                logits = policy(obs_t)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                logprob = dist.log_prob(actions)
                entropy = dist.entropy()
                values = value_net(obs_t).squeeze(1)

                next_obs, rewards, terminated, truncated, _ = envs.step(actions.cpu().numpy())
                dones = np.logical_or(terminated, truncated)

                obs_buf.append(obs_t)
                actions_buf.append(actions)
                logprob_buf.append(logprob)
                rewards_buf.append(torch.tensor(rewards, dtype=torch.float32, device=device))
                dones_buf.append(torch.tensor(dones.astype(np.float32), dtype=torch.float32, device=device))
                values_buf.append(values)
                entropy_buf.append(entropy)

                ep_rewards += rewards
                ep_lengths += 1

                for idx in np.where(dones)[0]:
                    episode_count += 1
                    total_reward = float(ep_rewards[idx])
                    episode_length = int(ep_lengths[idx])
                    rewards_history.append(total_reward)
                    ma = moving_average(rewards_history, window=50)[-1]
                    last_finished_reward = total_reward
                    last_finished_length = episode_length

                    ep_rewards[idx] = 0.0
                    ep_lengths[idx] = 0

                    if total_reward > best_reward:
                        best_reward = total_reward
                        torch.save(policy.state_dict(), model_dir / f"{args.exp_name}_best.pt")
                        torch.save(value_net.state_dict(), model_dir / f"{args.exp_name}_value_best.pt")

                    if episode_count % 10 == 0:
                        print(
                            f"[A2C] ep={episode_count:4d} reward={total_reward:8.2f} ma50={ma:8.2f} step={global_step:7d}"
                        )

            with torch.no_grad():
                next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
                next_values = value_net(next_obs_t).squeeze(1)

            rewards_t = torch.stack(rewards_buf)
            dones_t = torch.stack(dones_buf)
            values_t = torch.stack(values_buf)
            entropy_t = torch.stack(entropy_buf)
            logprob_t = torch.stack(logprob_buf)

            advantages = torch.zeros_like(rewards_t, device=device)
            last_gae = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
            for t in reversed(range(args.n_steps)):
                if t == args.n_steps - 1:
                    next_non_terminal = 1.0 - dones_t[t]
                    next_value = next_values
                else:
                    next_non_terminal = 1.0 - dones_t[t]
                    next_value = values_t[t + 1]

                delta = rewards_t[t] + args.gamma * next_value * next_non_terminal - values_t[t]
                last_gae = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae

            returns = advantages + values_t

            b_logprob = logprob_t.reshape(-1)
            b_values = values_t.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_entropy = entropy_t.reshape(-1)

            policy_loss = -(b_logprob * b_advantages.detach()).mean()
            value_loss = F.mse_loss(b_values, b_returns.detach())
            entropy = b_entropy.mean()

            total_loss = policy_loss + args.value_loss_coef * value_loss - args.entropy_coef * entropy

            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            total_loss.backward()

            policy_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            value_grad_norm = torch.nn.utils.clip_grad_norm_(value_net.parameters(), args.max_grad_norm)
            grad_norm = float(max(float(policy_grad_norm), float(value_grad_norm)))

            policy_optimizer.step()
            value_optimizer.step()

            if len(rewards_history) > 0:
                logger.log(
                    {
                        "episode": episode_count,
                        "total_reward": last_finished_reward,
                        "moving_avg_reward": moving_average(rewards_history, window=50)[-1],
                        "episode_length": last_finished_length,
                        "loss": float(total_loss.item()),
                        "learning_rate": policy_optimizer.param_groups[0]["lr"],
                        "seed": args.seed,
                        "exp_name": args.exp_name,
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy.item()),
                        "adv_mean": float(b_advantages.mean().item()),
                        "grad_norm": grad_norm,
                    }
                )

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
            "total_timesteps": args.total_timesteps,
            "episodes_finished": episode_count,
            "best_episode_reward": best_reward,
            "final_moving_avg_50": moving_average(rewards_history, 50)[-1] if rewards_history else None,
            "hyperparameters": vars(args),
        },
    )

    envs.close()
    print(f"Saved A2C policy to {final_policy_path}")


if __name__ == "__main__":
    main()
