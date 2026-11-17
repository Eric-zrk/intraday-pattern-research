import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="K-Shape + GRU/MLP intraday pattern research")
    subs = p.add_subparsers(dest="command", required=True)
    d = subs.add_parser("demo-data", help="Generate synthetic bars; not real market data")
    d.add_argument("--output", required=True)
    d.add_argument("--days", type=int, default=90)
    d.add_argument("--seed", type=int, default=42)
    r = subs.add_parser("run", help="Fit clusters, features and model with chronological holdouts")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--train-end", required=True)
    r.add_argument("--val-end", required=True)
    r.add_argument("--bar-label", choices=["start", "end"], default="start")
    r.add_argument("--model", choices=["hybrid", "gru", "mlp"], default="hybrid")
    r.add_argument("--epochs", type=int, default=100)
    r.add_argument("--batch-size", type=int, default=256)
    r.add_argument("--patience", type=int, default=10)
    r.add_argument("--lr", type=float, default=.001)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--cluster-iter", type=int, default=100)
    r.add_argument("--synthetic", action="store_true", help="Mark outputs as synthetic")
    i = subs.add_parser("predict", help="Predict from 20 prior sessions and current 10 bars")
    i.add_argument("--checkpoint", required=True)
    i.add_argument("--history", required=True, help="CSV/Parquet containing 20 complete prior sessions")
    i.add_argument("--opening", required=True, help="CSV containing exactly 10 current opening bars")
    i.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "demo-data":
        if args.days < 23:
            p.error("--days must be at least 23")
        from .demo import generate
        generate(args.output, args.days, args.seed)
    elif args.command == "run":
        if min(args.epochs, args.batch_size, args.patience, args.cluster_iter) < 1 or args.lr <= 0:
            p.error("Training parameters must be positive")
        from .experiment import run
        run(args)
    else:
        from .inference import predict
        predict(args)


if __name__ == "__main__":
    main()
