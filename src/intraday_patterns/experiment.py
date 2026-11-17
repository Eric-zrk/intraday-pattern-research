"""Train/validation/test experiment with train-only preprocessing and checkpoints."""
import copy
import json
import platform
import random
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
from .data import FEATURES, load_days, build_dataset, time_masks
from .clustering import fit_labels
from .model import PatternPredictor


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def evaluate(y, pred):
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            "class_counts": np.bincount(y, minlength=3).tolist()}


def probabilities(model, seq, cross, batch_size=256):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(seq[i:i+batch_size], dtype=torch.float32),
                                      torch.tensor(cross[i:i+batch_size], dtype=torch.float32)
                                   ).softmax(dim=1).numpy()
                               for i in range(0, len(seq), batch_size)])


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier experiment.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    days, audit, symbol = load_days(args.input, args.bar_label)
    args.synthetic = args.synthetic or symbol.upper().startswith("SYNTHETIC")
    audit.to_csv(out / "data_quality.csv", index=False)
    data = build_dataset(days)
    train, val, test = time_masks(data["dates"], args.train_end, args.val_end)
    labels, centers, cluster_info = fit_labels(data["shapes"], train, args.seed, args.cluster_iter)
    np.save(out / "cluster_centers.npy", centers)
    write_json(out / "clustering.json", cluster_info)
    mean = data["cross"][train].mean(axis=0)
    std = data["cross"][train].std(axis=0)
    std = np.where(std < 1e-12, 1., std)
    cross = (data["cross"] - mean) / std
    if not np.isfinite(cross).all():
        raise ValueError("Non-finite feature values")
    seq = data["seq"]
    loader = DataLoader(TensorDataset(torch.tensor(seq[train]), torch.tensor(cross[train]),
                                      torch.tensor(labels[train], dtype=torch.long)),
                        batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    model = PatternPredictor(args.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    best, best_state, stale, history, best_epoch = float("inf"), None, 0, [], 0
    for epoch in range(args.epochs):
        model.train()
        total = 0.
        for x, c, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x, c), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        vp = probabilities(model, seq[val], cross[val])
        vl = float(-np.log(np.clip(vp[np.arange(val.sum()), labels[val]], 1e-12, 1)).mean())
        history.append({"epoch": epoch + 1, "train_loss": total / train.sum(),
                        "val_loss": vl, "val_accuracy": float((vp.argmax(1) == labels[val]).mean())})
        if vl < best:
            best, best_state, best_epoch, stale = vl, copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            stale += 1
        print(f"epoch={epoch+1} train_loss={total/train.sum():.4f} val_loss={vl:.4f}", flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    checkpoint = {"state_dict": best_state, "mode": args.model, "features": FEATURES,
                  "cross_mean": mean.tolist(), "cross_std": std.tolist(),
                  "centers": centers.tolist(), "best_epoch": best_epoch,
                  "bar_label": args.bar_label, "symbol": symbol, "seed": args.seed,
                  "train_end": args.train_end, "val_end": args.val_end,
                  "synthetic": args.synthetic}
    torch.save(checkpoint, out / "model.pt")
    majority = int(np.bincount(labels[train], minlength=3).argmax())
    metrics = {"synthetic": args.synthetic, "model": args.model, "best_epoch": best_epoch,
               "train_n": int(train.sum()), "validation": evaluate(labels[val], probabilities(model, seq[val], cross[val]).argmax(1)),
               "uniform_random_expected_accuracy": 1 / 3}
    # Final test evaluation occurs only after model selection has finished.
    tp = probabilities(model, seq[test], cross[test])
    metrics["test"] = evaluate(labels[test], tp.argmax(1))
    metrics["majority_baseline_test"] = evaluate(labels[test], np.full(test.sum(), majority))
    pd.DataFrame({"date": data["dates"][test], "label": labels[test], "prediction": tp.argmax(1),
                  "prob_up": tp[:, 0], "prob_down": tp[:, 1], "prob_range": tp[:, 2]}).to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame({"date": data["dates"], "label": labels,
                  "split": np.where(train, "train", np.where(val, "validation", "test"))}).to_csv(out / "labels.csv", index=False)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    write_json(out / "metrics.json", metrics)
    write_json(out / "config.json", {**vars(args), "input": str(Path(args.input).name),
                                     "output": out.name, "symbol": symbol, "features": FEATURES})
    write_json(out / "environment.json", {"python": platform.python_version(), **{
        x: version(x) for x in ["numpy", "pandas", "torch", "tslearn", "scikit-learn", "scipy", "matplotlib", "pyarrow"]}})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.plot(centers[i, :, 0])
        ax.set_title(["Up prototype match", "Down prototype match", "Range prototype match"][i])
    fig.suptitle("SYNTHETIC DEMO - NOT MARKET EVIDENCE" if args.synthetic else "Training-only K-Shape centers: inspect semantics")
    fig.tight_layout()
    fig.savefig(out / "cluster_centers.png", dpi=150)
    plt.close(fig)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
