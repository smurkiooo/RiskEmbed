"""Checkpoint selection, customer isolation, padding and inference consistency."""
import numpy as np
import pytest
import torch

from riskembed.encoder.model import CausalSequenceEncoder, EncoderConfig
from riskembed.encoder.statements import StatementStore, StatementTokenizer, TabularVocab, collate_statements
from riskembed.evaluation.metrics import auc
from riskembed.models.supervised import StatementClassifier, SupervisedConfig, fit_classifier, predict_classifier


def tiny_classifier():
    torch.manual_seed(4)
    vocab = TabularVocab(["P_2"], [], [[0.5]], [])
    encoder = CausalSequenceEncoder(StatementTokenizer(vocab, 8),
                                    EncoderConfig(d_model=8, n_heads=2, n_layers=1, emb_dim=4, dropout=0.0))
    store = StatementStore(np.arange(24), np.arange(0, 49, 2), np.tile([19000, 19030], 24),
                           np.tile([[1], [2], [2], [1]], (12, 1)).astype(np.uint8), np.empty((48, 0), dtype=np.int16))
    return StatementClassifier(encoder), store


def test_selected_checkpoint_and_serialization(tmp_path):
    model, store = tiny_classifier()
    accessed = []
    original = store.client

    def client(i):
        accessed.append(i)
        return original(i)

    store.client = client
    train, valid = np.arange(12), np.arange(12, 20)
    y_train, y_valid = train % 2, valid % 2
    info = fit_classifier(model, store, train, y_train, valid, y_valid,
                          SupervisedConfig(epochs=3, batch_size=4, patience=2), log=lambda _: None)
    assert set(accessed).issubset(set(np.concatenate([train, valid])))  # test clients 20..23 were never read
    score = predict_classifier(model, store, valid)
    assert auc(y_valid, score) == pytest.approx(info["best_valid_auc"])
    assert info["best_valid_auc"] == max(r["valid_auc"] for r in info["history"])
    torch.save(model.state_dict(), tmp_path / "model.pt")
    restored, _ = tiny_classifier()
    restored.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    np.testing.assert_array_equal(score, predict_classifier(restored, store, valid))
    with pytest.raises(ValueError, match="overlap"):
        fit_classifier(model, store, train, y_train, train, y_train, SupervisedConfig())


def test_padding_does_not_change_last_real_prediction():
    model, store = tiny_classifier()
    model.eval()
    short = store.client(0)
    long = store.client(1)
    long = {**long, "days": np.array([19000,19030,19060]),
            "num_codes": np.array([[1],[2],[1]], dtype=np.uint8), "cat_idx": np.empty((3,0),dtype=np.int16)}
    with torch.no_grad():
        alone = model(collate_statements([short]))[0]
        padded = model(collate_statements([short,long]))[0]
    assert torch.allclose(alone, padded, atol=1e-6)
