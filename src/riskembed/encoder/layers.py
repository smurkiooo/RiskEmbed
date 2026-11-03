"""Блоки энкодера: календарное кодирование времени, причинное внимание со штрафом по Δt."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CalendarEncoding(nn.Module):
    """Фиксированные Фурье-признаки календаря: sin/cos фазы недели и месяца (по две гармоники).

    Это то, что в поведении клиента действительно циклично: выходные и день зарплаты. Нет линейного
    тренда и обучаемых частот (как в Time2Vec, arXiv:1907.05321) и нет годовой фазы: иначе эмбеддинг
    зависит от даты или сезона и дрейфует просто с течением времени. Оба периода делят 3 409 дней
    (7 × 487), поэтому сдвиг истории на это число дней не меняет кодирование — это проверяется тестом.
    """

    PERIODS_DAYS = (7.0, 365.25 / 12)
    HARMONICS = (1, 2)
    INVARIANT_SHIFT_DAYS = 3_409
    _SCALE = 32  # периоды × 32 — целые числа: фаза считается точно в int64 (MPS не поддерживает float64)

    def __init__(self):
        super().__init__()
        periods = [round(p / h * self._SCALE) for p in self.PERIODS_DAYS for h in self.HARMONICS]
        self.register_buffer("periods_scaled", torch.tensor(periods, dtype=torch.long), persistent=False)
        self.dim = 2 * len(periods)

    def forward(self, t_days: torch.Tensor) -> torch.Tensor:
        ticks = torch.remainder(t_days.long().unsqueeze(-1) * self._SCALE, self.periods_scaled)
        phase = 2 * math.pi * ticks.float() / self.periods_scaled.float()
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class TimeBiasedCausalAttention(nn.Module):
    """Причинное внимание со штрафом по Δt в духе ALiBi (arXiv:2108.12409).

    logits_ij = q_i·k_j/√d − m_h · log(1 + (t_i − t_j)), j ≤ i; наклоны m_h ≥ 0 обучаемы.
    time_bias=False — обычное причинное внимание (вариант для абляции).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, time_bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.time_bias = time_bias
        if time_bias:
            slopes = torch.tensor([2 ** (-8 * (i + 1) / n_heads) for i in range(n_heads)]) * 4
            self.raw_slope = nn.Parameter(torch.log(torch.expm1(slopes)))  # softplus^-1

    def slopes(self) -> torch.Tensor:
        return F.softplus(self.raw_slope)

    def forward(self, x: torch.Tensor, t_days: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """x [B,T,d]; t_days [B,T]; pad [B,T] True = паддинг."""
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)

        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        allowed = causal[None] & ~pad[:, None, :]  # [B,T,T]: j ≤ i и j не паддинг
        allowed = allowed | torch.eye(T, dtype=torch.bool, device=x.device)[None]  # паддинг-строки не дают NaN
        bias = torch.zeros(B, 1, T, T, device=x.device, dtype=x.dtype)
        bias = bias.masked_fill(~allowed[:, None], float("-inf"))
        if self.time_bias:
            dt = (t_days[:, :, None] - t_days[:, None, :]).clamp_min(0).to(x.dtype)  # [B,T,T]
            bias = bias - self.slopes().view(1, -1, 1, 1) * torch.log1p(dt)[:, None]

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    """Pre-LN блок трансформера. Нормализация — LayerNorm по токену: BatchNorm здесь
    недопустим, он смешивает статистики разных дат и клиентов (скрытая утечка из будущего)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, time_bias: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TimeBiasedCausalAttention(d_model, n_heads, dropout, time_bias)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, t_days, pad):
        x = x + self.drop(self.attn(self.ln1(x), t_days, pad))
        return x + self.drop(self.ff(self.ln2(x)))
