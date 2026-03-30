from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from src.common.utils import ensure_dir


def make_video_env(env_id: str, video_folder: Path | str, video_prefix: str, seed: int) -> gym.Env:
    """Create a RecordVideo wrapped env configured for a single episode."""
    video_folder = ensure_dir(video_folder)
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_folder),
        episode_trigger=lambda episode_id: True,
        name_prefix=f"{video_prefix}_seed{seed}",
        disable_logger=True,
    )
    return env
