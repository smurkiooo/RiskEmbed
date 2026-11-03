"""Score an integer-encoded AmEx parquet using a self-contained supervised model bundle."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from riskembed.data.amex import ID_COL, customer_int_id
from riskembed.encoder.model import CausalSequenceEncoder, EncoderConfig
from riskembed.encoder.statements import StatementTokenizer, TabularVocab, build_statement_store
from riskembed.models.supervised import StatementClassifier, predict_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(6)
    metadata = json.loads((args.model / "training.json").read_text())
    vocab = TabularVocab.load(args.model / "vocab.json")
    store, _ = build_statement_store(args.input, np.empty(0, dtype=np.int64), vocab=vocab)
    encoder = CausalSequenceEncoder(StatementTokenizer(vocab, metadata["encoder"]["d_model"]),
                                    EncoderConfig(**metadata["encoder"]))
    model = StatementClassifier(encoder)
    model.load_state_dict(torch.load(args.model / "model.pt", weights_only=True, map_location="cpu"))
    scores = predict_classifier(model, store, np.arange(len(store)))
    raw_ids = pq.read_table(args.input, columns=[ID_COL]).column(0).to_pandas().drop_duplicates()
    keys = pd.DataFrame({ID_COL: raw_ids.to_numpy(), "cid": customer_int_id(raw_ids)})
    out = keys.merge(pd.DataFrame({"cid": store.client_ids, "prediction": scores}),
                     on="cid", validate="one_to_one").drop(columns="cid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Scored {len(out):,} customers → {args.output}. Scores refer to the sampled dataset, not calibrated banking PD.")


if __name__ == "__main__":
    main()
