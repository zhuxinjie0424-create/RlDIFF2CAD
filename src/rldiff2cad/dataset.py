"""Dataset helpers for serialized sketch sequences stored as JSONL."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List

from .tokens import SketchSequence, Token, TokenKind, Vocabulary


@dataclass
class DatasetConfig:
    vocabulary: Vocabulary = Vocabulary()


def _token_from_dict(d: dict, vocab: Vocabulary) -> Token:
    token = Token(TokenKind(d["kind"]), d["value"])
    vocab.validate(token)
    return token


def load_jsonl(path: str | Path, config: DatasetConfig | None = None) -> Iterator[SketchSequence]:
    cfg = config or DatasetConfig()
    with Path(path).open() as f:
        for line in f:
            record = json.loads(line)
            seq = SketchSequence(vocabulary=cfg.vocabulary)
            seq.extend(_token_from_dict(t, cfg.vocabulary) for t in record["tokens"])
            yield seq


def batchify(sequences: Iterable[SketchSequence], batch_size: int = 4) -> Iterator[List[SketchSequence]]:
    batch: List[SketchSequence] = []
    for seq in sequences:
        batch.append(seq)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
