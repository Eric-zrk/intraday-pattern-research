"""Synthetic minute bars for exercising software, never for empirical conclusions."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import grid


def generate(path, n_days=90, seed=42):
    rng = np.random.default_rng(seed)
    rows, previous = [], 4000.
    for date in pd.bdate_range("2024-01-02", periods=n_days):
        t = np.linspace(0, 1, 240)
        pattern = rng.integers(3)
        trend = [12 * t, -12 * t, 6 * np.sin(8 * np.pi * t)][pattern]
        close = previous + rng.normal(0, 2) + trend + rng.normal(0, .6, 240)
        opening = np.r_[close[0], close[:-1]]
        volume = rng.integers(100, 1000, 240)
        rows.append(pd.DataFrame({"trade_time": grid(date), "symbol": "SYNTHETIC",
            "open": opening, "high": np.maximum(opening, close) + .2,
            "low": np.minimum(opening, close) - .2, "close": close,
            "volume": volume, "amount": volume * close * 200}))
        previous = close[-1]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    pd.concat(rows).to_csv(path, index=False)
