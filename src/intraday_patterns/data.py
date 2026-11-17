"""Strict minute-bar validation and causal, date-aligned feature engineering."""
from pathlib import Path
import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume", "amount"]
FEATURES = ["ret_open", "ret_open10", "amt_open10", "std_open10"] + [
    f"ret_{part}_p{lag}" for part in ("o10", "c10", "m10") for lag in range(1, 6)
] + ["skew_ret", "kurt_vol", "std_ret", "corr_rv", "zs_ret", "zs_amt"]


def grid(date):
    date = pd.Timestamp(date).normalize()
    return pd.date_range(date + pd.Timedelta(hours=9, minutes=30), periods=120, freq="min").append(
        pd.date_range(date + pd.Timedelta(hours=13), periods=120, freq="min"))


def zscore(x, axis=0):
    x = np.asarray(x, dtype=np.float64)
    std = x.std(axis=axis, keepdims=True)
    return (x - x.mean(axis=axis, keepdims=True)) / np.where(std < 1e-12, 1., std)


def load_days(path, bar_label="start"):
    """Require one explicitly identified instrument and complete 240-bar days.

    No backward filling or interpolation can introduce later-day data into inputs.
    End-labeled bars are moved back one minute to a common start-labeled grid.
    """
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    required = ["trade_time", "symbol"] + FIELDS
    if set(required) - set(df):
        raise ValueError(f"Missing columns: {sorted(set(required) - set(df))}")
    if df.symbol.isna().any() or df.symbol.nunique() != 1:
        raise ValueError("Use exactly one non-null symbol per input file.")
    if bar_label not in ("start", "end"):
        raise ValueError("bar_label must be start or end")
    df["trade_time"] = pd.to_datetime(df.trade_time, errors="raise")
    if df.trade_time.isna().any():
        raise ValueError("Null timestamps are not allowed")
    if df.trade_time.dt.tz is not None:
        raise ValueError("Convert timestamps to naive Asia/Shanghai wall time first.")
    if bar_label == "end":
        df["trade_time"] -= pd.Timedelta(minutes=1)
    if df.trade_time.duplicated().any():
        raise ValueError("Duplicate timestamps; resolve overlapping source files explicitly.")
    df[FIELDS] = df[FIELDS].apply(pd.to_numeric, errors="raise")
    df = df.sort_values("trade_time").set_index("trade_time")
    days, audit = {}, []
    # Keep all observed dates, including incomplete ones, to prevent bridging rejected days.
    for date, g in df.groupby(df.index.normalize()):
        day = g.reindex(grid(date))[FIELDS]
        a = day.to_numpy(dtype=float)
        valid = np.isfinite(a).all()
        valid &= (a[:, :4] > 0).all() and (a[:, 4:] >= 0).all()
        valid &= (a[:, 1] >= a[:, [0, 2, 3]].max(axis=1)).all()
        valid &= (a[:, 2] <= a[:, [0, 1, 3]].min(axis=1)).all()
        valid &= np.std(a[:, 3]) > 1e-12
        days[date] = a if valid else None
        audit.append({"date": str(date.date()), "accepted": bool(valid),
                      "missing_bars": int(day.isna().any(axis=1).sum()),
                      "outside_session_rows": int((~g.index.isin(grid(date))).sum())})
    if not any(a is not None for a in days.values()):
        raise ValueError("No complete, valid 240-bar days. Check bar-label convention.")
    return days, pd.DataFrame(audit), str(df.symbol.iloc[0])


def safe_corr(a, b):
    return 0. if np.std(a) < 1e-12 or np.std(b) < 1e-12 else float(np.corrcoef(a, b)[0, 1])


def anomaly(x):
    return 0. if np.std(x) < 1e-12 else float((x[-1] - np.mean(x)) / np.std(x))


def sample(history, current_open10):
    """Use exactly 20 prior full sessions and the first 10 completed current bars.

    This API intentionally never receives the current day's remaining bars.
    """
    if len(history) != 20 or np.shape(current_open10) != (10, 6):
        raise ValueError("Expected 20 history days and current_open10 shape (10, 6)")
    h = np.asarray(history)
    now = np.asarray(current_open10)
    seq = zscore(np.concatenate([h[-1, -230:], now]), axis=0)
    open10 = h[:, 9, 3] / h[:, 0, 0] - 1
    close10 = h[:, -1, 3] / h[:, -10, 0] - 1
    # Last 5 morning bars + first 5 afternoon bars; a defined convention, not lunch trades.
    middle10 = h[:, 124, 3] / h[:, 115, 0] - 1
    returns = h[:, -1, 3] / h[:, 0, 0] - 1
    volumes = h[:, :, 4].sum(axis=1)
    amounts = h[:, :, 5].sum(axis=1)
    r5, v5 = returns[-5:], volumes[-5:]
    zr, zv = zscore(r5), zscore(v5)
    cross = [now[0, 0] / h[-1, -1, 3] - 1, now[-1, 3] / now[0, 0] - 1,
             now[:, 5].sum(), np.std(np.diff(np.log(now[:, 3])), ddof=0)]
    cross += [arr[-lag] for arr in (open10, close10, middle10) for lag in range(1, 6)]
    cross += [np.mean(zr ** 3), np.mean(zv ** 4) - 3, np.std(r5), safe_corr(r5, v5),
              anomaly(returns), anomaly(amounts)]
    return seq.astype(np.float32), np.asarray(cross, dtype=np.float32)


def build_dataset(days):
    dates, seqs, crosses, targets = [], [], [], []
    keys = sorted(days)
    for i in range(20, len(keys)):
        window = [days[d] for d in keys[i-20:i+1]]
        if any(a is None for a in window):
            continue
        seq, cross = sample(window[:-1], window[-1][:10])
        dates.append(str(keys[i].date()))
        seqs.append(seq)
        crosses.append(cross)
        targets.append(zscore(window[-1][:, 3]))
    if not dates:
        raise ValueError("No samples after the 20-session warmup and quality filtering.")
    return {"dates": np.asarray(dates), "seq": np.stack(seqs), "cross": np.stack(crosses),
            "shapes": np.stack(targets)[:, :, None]}


def time_masks(dates, train_end, val_end):
    dates = np.asarray(dates, dtype="datetime64[D]")
    train_end, val_end = np.datetime64(train_end), np.datetime64(val_end)
    if train_end >= val_end:
        raise ValueError("train_end must be strictly before val_end")
    masks = [dates <= train_end, (dates > train_end) & (dates <= val_end), dates > val_end]
    if any(not m.any() for m in masks):
        raise ValueError("Train, validation and test must all be non-empty after warmup.")
    if masks[0].sum() < 3:
        raise ValueError("K=3 requires at least three training samples.")
    return masks
