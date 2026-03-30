from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from src.common.logger import CSVLogger
from src.common.seed import set_global_seed
from src.common.utils import ensure_dir, get_device, moving_average, save_json
from src.dqn.model import QNetwork
from src.dqn.replay_buffer import ReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on LunarLander-v3")
    parser.add_argument("--exp-name", type=str, default="dqn_default")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=2000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--target-update-freq", type=int, default=250)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--tau", type=float, default=1.0, help="1.0 = hard update")
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

    q_net = QNetwork(state_dim, action_dim, args.hidden_size).to(device)
    target_net = QNetwork(state_dim, action_dim, args.hidden_size).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay_buffer = ReplayBuffer(args.buffer_size)

    run_dir = ensure_dir(Path("logs/dqn") / args.exp_name)
    model_dir = ensure_dir(Path("models/dqn"))
    metrics_file = run_dir / "metrics.csv"

    fieldnames = [
        "episode",
        "total_reward",
        "moving_avg_reward",
        "episode_length",
        "loss",
        "learning_rate",
        "seed",
        "exp_name",
        "epsilon",
        "buffer_size",
        "mean_q",
    ]

    global_step = 0
    epsilon = args.epsilon_start
    rewards_history: list[float] = []
    best_reward = -float("inf")

    with CSVLogger(metrics_file, fieldnames) as logger:
        for episode in range(1, args.episodes + 1):
            state, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            losses = []
            q_vals = []

            for step in range(args.max_steps):
                global_step += 1
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        q_value = q_net(st)
                        action = int(torch.argmax(q_value, dim=1).item())
                        q_vals.append(float(q_value.max().item()))

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay_buffer.add(state, action, reward, next_state, float(done))
                total_reward += reward
                state = next_state

                if (
                    len(replay_buffer) >= args.batch_size
                    and global_step >= args.learning_starts
                    and global_step % args.train_freq == 0
                ):
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(args.batch_size)
                    states = torch.tensor(b_s, dtype=torch.float32, device=device)
                    actions = torch.tensor(b_a, dtype=torch.int64, device=device).unsqueeze(1)
                    rewards = torch.tensor(b_r, dtype=torch.float32, device=device).unsqueeze(1)
                    next_states = torch.tensor(b_ns, dtype=torch.float32, device=device)
                    dones = torch.tensor(b_d, dtype=torch.float32, device=device).unsqueeze(1)

                    q_pred = q_net(states).gather(1, actions)
                    with torch.no_grad():
                        q_next = target_net(next_states).max(dim=1, keepdim=True).values
                        q_target = rewards + args.gamma * (1.0 - dones) * q_next

                    loss = F.mse_loss(q_pred, q_target)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                    optimizer.step()
                    losses.append(float(loss.item()))

                if global_step % args.target_update_freq == 0:
                    if args.tau >= 1.0:
                        target_net.load_state_dict(q_net.state_dict())
                    else:
                        for tgt, src in zip(target_net.parameters(), q_net.parameters()):
                            tgt.data.copy_(args.tau * src.data + (1.0 - args.tau) * tgt.data)

                if done:
                    break

            epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
            rewards_history.append(total_reward)
            ma = moving_average(rewards_history, window=50)[-1]
            avg_loss = float(np.mean(losses)) if losses else 0.0
            mean_q = float(np.mean(q_vals)) if q_vals else 0.0

            logger.log(
                {
                    "episode": episode,
                    "total_reward": total_reward,
                    "moving_avg_reward": ma,
                    "episode_length": step + 1,
                    "loss": avg_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "seed": args.seed,
                    "exp_name": args.exp_name,
                    "epsilon": epsilon,
                    "buffer_size": len(replay_buffer),
                    "mean_q": mean_q,
                }
            )

            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(q_net.state_dict(), model_dir / f"{args.exp_name}_best.pt")

            if episode % 25 == 0:
                print(
                    f"[DQN] ep={episode:4d} reward={total_reward:8.2f} "
                    f"ma50={ma:8.2f} eps={epsilon:.3f}"
                )

    final_model_path = model_dir / f"{args.exp_name}.pt"
    torch.save(q_net.state_dict(), final_model_path)
    save_json(
        run_dir / "summary.json",
        {
            "algorithm": "dqn",
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
    print(f"Saved DQN model to {final_model_path}")


if __name__ == "__main__":
    main()
