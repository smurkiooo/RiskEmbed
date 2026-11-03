"""Negative control on real quick-mode histories: independently shuffled labels in every split.

A near-random result checks the training/evaluation path, not semantic leakage in original source features.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from riskembed.config import load_config, resolve
from riskembed.encoder.model import CausalSequenceEncoder, EncoderConfig
from riskembed.encoder.statements import StatementStore, StatementTokenizer, TabularVocab
from riskembed.evaluation.metrics import auc
from riskembed.evaluation.splits import holdout_mask
from riskembed.models.supervised import StatementClassifier, SupervisedConfig, fit_classifier, predict_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['quick', 'full'], default='quick')
    args = parser.parse_args()
    cfg = load_config()
    work = resolve(cfg.amex.work_dir) / args.source
    labels = pd.read_parquet(work/'labels.parquet')
    if args.source == 'full':
        labels = labels[holdout_mask(labels.cid,0.03,salt=99)].reset_index(drop=True)
    store=StatementStore.load(work/'statements')
    vocab=TabularVocab.load(work/'vocab.json')
    positions=store.index_of()
    idx=np.array([positions[int(c)] for c in labels.cid])
    rows={s:np.flatnonzero(labels.split.to_numpy()==s) for s in ['train','valid','test']}
    torch.set_num_threads(2)
    results=[]
    for seed in [7,42,101]:
        rng=np.random.default_rng(seed)
        y={s:rng.permutation(labels.target.to_numpy()[r]) for s,r in rows.items()}
        torch.manual_seed(seed)
        encoder=CausalSequenceEncoder(StatementTokenizer(vocab,cfg.amex.encoder.d_model),EncoderConfig(**dict(cfg.amex.encoder)))
        model=StatementClassifier(encoder)
        trained=fit_classifier(model,store,idx[rows['train']],y['train'],idx[rows['valid']],y['valid'],
                               SupervisedConfig(epochs=2,batch_size=128,seed=seed),log=lambda _:None)
        prediction=predict_classifier(model,store,idx[rows['test']])
        results.append({'seed':seed,'validation_auc':trained['best_valid_auc'],'test_auc':auc(y['test'],prediction)})
        print(results[-1],flush=True)
    report={'mode':'control_subset', 'source':args.source,'clients':len(labels),'runs':results,
            'mean_test_auc':float(np.mean([r['test_auc'] for r in results])),
            'interpretation':'Labels are independently permuted within all splits. Near 0.5 supports a functional negative control.',
            'limitation':'This does not rule out original-feature proxies for future outcomes or establish temporal generalization.'}
    (resolve(cfg.reports_dir)/'label_shuffle_control.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
