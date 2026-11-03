"""Compare the same Transformer trained from scratch and fine-tuned after CoLES+VICReg."""
import argparse
import gc
import json
import time

import numpy as np
import pandas as pd
import torch

from riskembed.config import load_config, resolve
from riskembed.data.amex import amex_metric
from riskembed.encoder.model import CausalSequenceEncoder, EncoderConfig
from riskembed.encoder.statements import StatementStore, StatementTokenizer, TabularVocab
from riskembed.evaluation.metrics import gini_bootstrap, paired_gini_delta_bootstrap
from riskembed.models.supervised import (
    StatementClassifier,
    SupervisedConfig,
    fit_classifier,
    predict_classifier,
)
from riskembed.provenance import file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    a = cfg.amex
    tag, suffix = ("quick", "_quick") if args.quick else ("full", "")
    work, art = resolve(a.work_dir) / tag, resolve("artifacts") / f"amex_{tag}"
    out = resolve(cfg.reports_dir) / f"amex_supervised{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(work / "labels.parquet")
    store = StatementStore.load(work / "statements")
    vocab = TabularVocab.load(work / "vocab.json")
    position = store.index_of()
    indices = np.array([position[int(cid)] for cid in labels.cid], dtype=np.int64)
    rows = {s: np.flatnonzero(labels.split.to_numpy() == s) for s in ("train", "valid", "test")}
    y = labels.target.to_numpy(dtype=np.float32)
    parameters = dict(a.supervised)
    if args.quick:
        parameters |= {"epochs": 2, "batch_size": 128}
    torch.set_num_threads(6)
    results, predictions = [], {}
    for name, pretrained in (("transformer_scratch", False), ("transformer_finetuned", True)):
        started = time.monotonic()
        torch.manual_seed(cfg.seed)
        encoder = CausalSequenceEncoder(StatementTokenizer(vocab, a.encoder.d_model), EncoderConfig(**dict(a.encoder)))
        if pretrained:
            encoder.load_state_dict(torch.load(art / "coles_vicreg" / "encoder.pt", weights_only=True, map_location="cpu"))
        model = StatementClassifier(encoder)
        train_cfg = SupervisedConfig(**parameters, seed=cfg.seed)
        if pretrained:
            train_cfg.encoder_lr_mult = 0.3
        print(f"{name}: train={len(rows['train'])}, valid={len(rows['valid'])}", flush=True)
        training = fit_classifier(model, store, indices[rows["train"]], y[rows["train"]],
                                  indices[rows["valid"]], y[rows["valid"]], train_cfg,
                                  log=lambda message: print(message, flush=True))
        destination = art / name
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), destination / "model.pt")
        vocab.save(destination / "vocab.json")
        # The exact tokenizer and architecture are bundled with the trained classifier.
        metadata = {"encoder": dict(a.encoder), "pretrained": pretrained, "seed": cfg.seed,
                    "labels_sha256": file_sha256(work / "labels.parquet"),
                    "vocab_sha256": file_sha256(work / "vocab.json"), **training}
        (destination / "training.json").write_text(json.dumps(metadata, indent=2))
        test_rows = rows["test"]
        scores = predict_classifier(model, store, indices[test_rows], train_cfg.batch_size)
        predictions[name] = scores
        scores_frame = labels.iloc[test_rows][["cid", "split", "target"]].copy()
        scores_frame["score"] = scores
        scores_frame.to_parquet(out / f"{name}_test.parquet", index=False)
        bootstrap = gini_bootstrap(y[test_rows], scores, labels.cid.to_numpy()[test_rows], a.bootstrap_reps, cfg.seed)
        result = {"name": name, "gini_test": bootstrap["gini"],
                  "gini_test_ci_low": bootstrap["ci_low"], "gini_test_ci_high": bootstrap["ci_high"],
                  "gini_valid": 2 * training["best_valid_auc"] - 1,
                  "best_epoch": training["best_epoch"], "parameters": sum(p.numel() for p in model.parameters()),
                  "seconds": round(time.monotonic() - started, 2), **amex_metric(y[test_rows], scores)}
        results.append(result)
        (out / f"{name}_training.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(result), flush=True)
        del model, encoder
        gc.collect()
    delta = paired_gini_delta_bootstrap(y[rows["test"]], predictions["transformer_scratch"],
                                        predictions["transformer_finetuned"], labels.cid.to_numpy()[rows["test"]],
                                        a.bootstrap_reps, cfg.seed)
    results[1] |= {"delta_vs_scratch": delta["delta"], "delta_scratch_ci_low": delta["ci_low"],
                   "delta_scratch_ci_high": delta["ci_high"]}
    pd.DataFrame(results).to_csv(out / "ablation.csv", index=False)
    print(out / "ablation.csv", flush=True)


if __name__ == "__main__":
    main()
