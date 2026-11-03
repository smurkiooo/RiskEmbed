"""Самообучение энкодера: CoLES (InfoNCE по подотрезкам) + VICReg."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from riskembed.encoder.model import CausalSequenceEncoder, last_token


@dataclass
class PretrainConfig:
    epochs: int = 3
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    min_crop: int = 3  # минимальная длина подотрезка в токенах
    max_crop: int = 13
    temperature: float = 0.1
    vicreg_var: float = 1.0  # 0 → «CoLES без VICReg» (строка абляции)
    vicreg_cov: float = 1.0
    grad_clip: float = 1.0
    min_len: int = 2  # клиенты короче не участвуют (две разные «картинки» из одного токена не построить)
    max_steps_per_epoch: int | None = None
    max_tokens: int | None = None  # ограничение B × T_max (ограничение памяти внимания)
    seed: int = 0
    device: str = "cpu"

    def to_dict(self) -> dict:
        return asdict(self)


def sample_crop(n: int, rng: np.random.Generator, min_len: int, max_len: int) -> tuple[int, int]:
    """Случайный отрезок [a, b) длины L ~ U[min(min_len, n), min(max_len, n)]."""
    hi = min(max_len, n)
    lo = min(min_len, hi)
    length = int(rng.integers(lo, hi + 1))
    a = int(rng.integers(0, n - length + 1))
    return a, a + length


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Симметричный NT-Xent: для i-го элемента z1 позитив — i-й элемент z2, негативы — все остальные 2B−2."""
    z = F.normalize(torch.cat([z1, z2]), dim=-1)
    B = z1.shape[0]
    sim = z @ z.T / temperature
    sim = sim.masked_fill(torch.eye(2 * B, dtype=torch.bool, device=z.device), float("-inf"))
    target = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, target)


def vicreg_terms(z: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """(variance-hinge, covariance) для батча эмбеддингов [B, k]."""
    zc = z - z.mean(dim=0)
    std = torch.sqrt(zc.var(dim=0) + eps)
    var_loss = F.relu(1.0 - std).mean()
    B, k = z.shape
    cov = (zc.T @ zc) / max(B - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off**2).sum() / k
    return var_loss, cov_loss


def embedding_diagnostics(emb: np.ndarray) -> dict[str, float]:
    """Качество пространства эмбеддингов без меток: схлопывание, корреляция, эффективный ранг."""
    emb = emb[np.isfinite(emb).all(axis=1)]
    std = emb.std(axis=0)
    corr = np.corrcoef(emb, rowvar=False)
    k = corr.shape[0]
    off = np.abs(corr[~np.eye(k, dtype=bool)])
    eig = np.clip(np.linalg.eigvalsh(np.cov(emb, rowvar=False)), 0, None)
    p = eig / max(eig.sum(), 1e-12)
    eff_rank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    return {
        "std_mean": float(std.mean()),
        "std_min": float(std.min()),
        "collapsed_dims_share": float((std < 0.05 * max(std.max(), 1e-12)).mean()),
        "abs_offdiag_corr_mean": float(np.nanmean(off)),
        "effective_rank": eff_rank,
        "dim": int(k),
    }


def pretrain(
    model: CausalSequenceEncoder,
    store,
    collate_fn: Callable,
    crop_fn: Callable,
    cfg: PretrainConfig,
    client_indices: np.ndarray | None = None,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Обучает model на подотрезках клиентов store. Метки не используются."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    lengths = np.diff(np.asarray(store.day_offsets))
    idx = np.arange(len(store)) if client_indices is None else np.asarray(client_indices)
    idx = idx[lengths[idx] >= cfg.min_len]
    steps_per_epoch = len(idx) // cfg.batch_size
    if cfg.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, cfg.max_steps_per_epoch)
    if steps_per_epoch < 1:
        raise ValueError("Not enough eligible train clients for one contrastive batch")
    total = max(cfg.epochs * steps_per_epoch, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total, pct_start=0.1)

    history = []
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        perm = rng.permutation(idx)
        agg = {"loss": 0.0, "nce": 0.0, "var": 0.0, "cov": 0.0}
        t0 = time.time()
        for s in range(steps_per_epoch):
            batch_idx = perm[s * cfg.batch_size : (s + 1) * cfg.batch_size]
            views = ([], [])
            for i in batch_idx:
                c = store.client(int(i))
                n = len(c["days"])
                max_crop = cfg.max_crop
                if cfg.max_tokens:  # ограничение памяти внимания: B × T ≤ max_tokens
                    max_crop = min(max_crop, max(cfg.max_tokens // (2 * cfg.batch_size), cfg.min_crop))
                for v in views:
                    a, b = sample_crop(n, rng, cfg.min_crop, max_crop)
                    v.append(crop_fn(c, a, b))
            b1 = collate_fn(views[0], device)
            b2 = collate_fn(views[1], device)
            z1 = last_token(model(b1), b1["lengths"])
            z2 = last_token(model(b2), b2["lengths"])

            nce = info_nce(z1, z2, cfg.temperature)
            v1, c1 = vicreg_terms(z1)
            v2, c2 = vicreg_terms(z2)
            var_l, cov_l = (v1 + v2) / 2, (c1 + c2) / 2
            loss = nce + cfg.vicreg_var * var_l + cfg.vicreg_cov * cov_l

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            for k_, v_ in (("loss", loss), ("nce", nce), ("var", var_l), ("cov", cov_l)):
                agg[k_] += float(v_.detach())
            if (s + 1) % max(steps_per_epoch // 5, 1) == 0:
                log(f"  epoch {epoch + 1} step {s + 1}/{steps_per_epoch} loss={agg['loss'] / (s + 1):.4f} "
                    f"nce={agg['nce'] / (s + 1):.4f} var={agg['var'] / (s + 1):.4f} cov={agg['cov'] / (s + 1):.4f}")
        rec = {"epoch": epoch + 1, **{k_: v_ / max(steps_per_epoch, 1) for k_, v_ in agg.items()},
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        log(f"epoch {epoch + 1}: {rec}")
    return history
