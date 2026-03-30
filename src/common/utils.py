from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_json(path: Path | str, data: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def moving_average(values: list[float], window: int = 50) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start : i + 1]) / (i - start + 1))
    return out
