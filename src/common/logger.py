from __future__ import annotations

import csv
from pathlib import Path

from src.common.utils import ensure_dir


class CSVLogger:
    """Minimal CSV logger with explicit headers."""

    def __init__(self, path: Path | str, fieldnames: list[str]) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.fieldnames = fieldnames
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def log(self, row: dict) -> None:
        safe_row = {k: row.get(k, "") for k in self.fieldnames}
        self._writer.writerow(safe_row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
