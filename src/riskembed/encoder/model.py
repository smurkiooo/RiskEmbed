"""каузальный энкодер: токенизатор --> трансформер с непрерывным временем."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from riskembed.encoder.layers import Block, CalendarEncoding


@dataclass
class EncoderConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    emb_dim: int = 32  # размерность выходного эмбеддинга клиента
    dropout: float = 0.1
    time_encoding: str = "continuous"  # continuous (календарь + штраф по Δt) | positions (абляция)
    max_positions: int = 2048

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSequenceEncoder(nn.Module):
    """Причинный трансформер над токенами любого токенизатора: tokenizer(batch, B, T) -> [B, T, d]."""

    def __init__(self, tokenizer: nn.Module, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tokenizer = tokenizer
        continuous = cfg.time_encoding == "continuous"
        if continuous:
            self.calendar = CalendarEncoding()
            self.time_proj = nn.Linear(self.calendar.dim, d)
            self.time_norm = nn.LayerNorm(d)  # вклад времени того же масштаба, что и токен
        elif cfg.time_encoding == "positions":
            self.pos = nn.Embedding(cfg.max_positions, d)
        else:
            raise ValueError(cfg.time_encoding)
        self.blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.dropout, time_bias=continuous) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.emb_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """→ эмбеддинги [B, T, emb_dim] для каждого токена (паддинг-позиции не использовать)."""
        B, T = batch["days"].shape
        x = self.tokenizer(batch, B, T)
        if self.cfg.time_encoding == "continuous":
            x = x + self.time_norm(self.time_proj(self.calendar(batch["days"])))
        else:
            pos = torch.arange(T, device=x.device).clamp_max(self.cfg.max_positions - 1)
            x = x + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x, batch["days"], batch["pad"])
        return self.head(self.ln_f(x))


def last_token(emb: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Эмбеддинг последнего реального токена каждой последовательности: [B, T, k] → [B, k]."""
    idx = (lengths.to(emb.device) - 1).clamp_min(0)
    return emb[torch.arange(emb.shape[0], device=emb.device), idx]


