"""Supervised sequence classification with validation-only checkpoint selection."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from riskembed.encoder.model import CausalSequenceEncoder, last_token
from riskembed.encoder.statements import collate_statements
from riskembed.evaluation.metrics import auc


@dataclass
class SupervisedConfig:
    epochs: int = 8
    batch_size: int = 512
    lr: float = 0.001
    encoder_lr_mult: float = 1.0
    weight_decay: float = 0.01
    patience: int = 2
    seed: int = 42


class StatementClassifier(nn.Module):
    def __init__(self, encoder: CausalSequenceEncoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.cfg.emb_dim, 1)

    def forward(self, batch):
        return self.classifier(last_token(self.encoder(batch), batch["lengths"])).squeeze(-1)


@torch.no_grad()
def predict_classifier(model, store, indices, batch_size=512):
    model.eval()
    predictions = []
    for start in range(0, len(indices), batch_size):
        batch = collate_statements([store.client(int(i)) for i in indices[start:start + batch_size]])
        predictions.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0)


def fit_classifier(model, store, train_indices, y_train, valid_indices, y_valid,
                   cfg: SupervisedConfig, log: Callable = print):
    """Only train/valid are accepted: test labels cannot affect optimization or early stopping."""
    if not len(train_indices) or not len(valid_indices):
        raise ValueError("Both train and validation must contain clients")
    if np.intersect1d(train_indices, valid_indices).size:
        raise ValueError("Train and validation customers overlap")
    if len(train_indices) != len(y_train) or len(valid_indices) != len(y_valid):
        raise ValueError("Targets must match sequence indices")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.lr * cfg.encoder_lr_mult},
        {"params": model.classifier.parameters(), "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_epoch, stale, best_state = -np.inf, 0, 0, None
    history = []
    for epoch in range(cfg.epochs):
        started = time.monotonic()
        model.train()
        permutation = rng.permutation(len(train_indices))
        total_loss = 0.0
        for start in range(0, len(permutation), cfg.batch_size):
            rows = permutation[start:start + cfg.batch_size]
            batch = collate_statements([store.client(int(i)) for i in train_indices[rows]])
            targets = torch.as_tensor(np.asarray(y_train)[rows], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(rows)
        valid_scores = predict_classifier(model, store, valid_indices, cfg.batch_size)
        valid_auc = auc(y_valid, valid_scores)
        if not np.isfinite(valid_auc):
            raise ValueError("Validation AUC is undefined; both classes are required")
        record = {"epoch": epoch + 1, "train_bce": total_loss / len(train_indices),
                  "valid_auc": valid_auc, "seconds": round(time.monotonic() - started, 2)}
        history.append(record)
        log(f"supervised epoch {epoch + 1}: {record}")
        if valid_auc > best_auc:
            best_auc, best_epoch, stale = valid_auc, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return {"config": asdict(cfg), "history": history, "best_epoch": best_epoch, "best_valid_auc": best_auc}
