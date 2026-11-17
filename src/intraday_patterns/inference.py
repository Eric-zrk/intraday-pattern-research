"""CPU inference with the same causal feature API as training."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import FEATURES, FIELDS, grid, load_days, sample
from .model import PatternPredictor
from .experiment import probabilities, write_json


def predict(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["features"] != FEATURES:
        raise ValueError("Checkpoint feature schema does not match this package")
    history, _, symbol = load_days(args.history, checkpoint["bar_label"])
    if symbol != checkpoint["symbol"]:
        raise ValueError("History symbol differs from checkpoint")
    if len(history) != 20 or any(v is None for v in history.values()):
        raise ValueError("History must contain exactly 20 complete observed trading sessions")
    opening = pd.read_csv(args.opening)
    required = ["trade_time", "symbol"] + FIELDS
    if set(required) - set(opening):
        raise ValueError("Opening file missing required columns")
    if len(opening) != 10 or not (opening.symbol == symbol).all():
        raise ValueError("Opening file must contain 10 bars of the checkpoint symbol")
    opening["trade_time"] = pd.to_datetime(opening.trade_time)
    if opening.trade_time.dt.tz is not None:
        raise ValueError("Use naive Asia/Shanghai timestamps")
    if checkpoint["bar_label"] == "end":
        opening["trade_time"] -= pd.Timedelta(minutes=1)
    opening = opening.sort_values("trade_time").set_index("trade_time")
    date = opening.index[0].normalize()
    if not opening.index.equals(grid(date)[:10]) or date <= max(history):
        raise ValueError("Opening timestamps must be the next session's first 10 minute bars")
    a = opening[FIELDS].to_numpy(dtype=float)
    if not np.isfinite(a).all() or (a[:, :4] <= 0).any() or (a[:, 4:] < 0).any():
        raise ValueError("Invalid opening prices/volume/amount")
    if (a[:, 1] < a[:, [0, 2, 3]].max(axis=1)).any() or (a[:, 2] > a[:, [0, 1, 3]].min(axis=1)).any():
        raise ValueError("Inconsistent opening OHLC bars")
    seq, cross = sample([history[k] for k in sorted(history)], a)
    cross = (cross - np.asarray(checkpoint["cross_mean"])) / np.asarray(checkpoint["cross_std"])
    model = PatternPredictor(checkpoint["mode"])
    model.load_state_dict(checkpoint["state_dict"])
    prob = probabilities(model, seq[None], cross[None])[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"date": str(date.date()), "symbol": symbol,
        "synthetic": checkpoint["synthetic"], "label": int(prob.argmax()),
        "probabilities": dict(zip(["up_prototype_match", "down_prototype_match", "range_prototype_match"], prob.tolist())),
        "available_after": str(date + pd.Timedelta(hours=9, minutes=40))})
    print(output)
