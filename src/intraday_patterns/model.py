import torch
from torch import nn


class PatternPredictor(nn.Module):
    """Hybrid GRU-MLP classifier with batch-size-one-safe LayerNorm."""
    def __init__(self, mode="hybrid"):
        super().__init__()
        if mode not in ("hybrid", "gru", "mlp"):
            raise ValueError("Unknown model mode")
        self.mode = mode
        self.gru = nn.GRU(6, 64, num_layers=2, batch_first=True, dropout=.3)
        self.cross = nn.Sequential(nn.Linear(25, 64), nn.LayerNorm(64), nn.ReLU(),
                                   nn.Dropout(.4), nn.Linear(64, 32), nn.LayerNorm(32),
                                   nn.ReLU(), nn.Dropout(.4))
        dim = {"hybrid": 96, "gru": 64, "mlp": 32}[mode]
        self.head = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Dropout(.3), nn.Linear(32, 3))

    def forward(self, seq, cross):
        parts = []
        if self.mode != "mlp":
            out, _ = self.gru(seq)
            parts.append(out[:, -1])
        if self.mode != "gru":
            parts.append(self.cross(cross))
        return self.head(torch.cat(parts, dim=1))
