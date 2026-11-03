"""Small file/config fingerprints to prevent silent reuse of stale experiment artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def experiment_signature(config: dict, files: list[Path]) -> str:
    payload = {"config": config, "inputs": {str(p): file_sha256(p) for p in files}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
