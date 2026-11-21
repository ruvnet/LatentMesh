"""Shared definitions for the preregistered contextual residual comparison."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
CONTEXT = HERE.parent / "context"
sys.path.insert(0, str(CONTEXT))
from develop import blob, predecessors
from corpus import describe, history, partition, sha
from development import fit_rows
from cohorts import ASSETS, Encoder, ensure, file_hash, json_write
from features import compact
from receiver import parameters as incumbent, unpack

PAIRS = [(j, (j + 1) % 8) for j in range(8)] + [(j, (j + 2) % 8) for j in range(4)]


def products(current, previous):
    return np.column_stack([current[:, j] * previous[:, k] for j, k in PAIRS])


def pair_state(encoder, previous, current):
    tokens = encoder.tokenizer.encode(previous or "", current)
    feed = {"input_ids": np.array([tokens.ids], dtype=np.int64),
            "attention_mask": np.array([tokens.attention_mask], dtype=np.int64)}
    return encoder.graph.run([encoder.hidden], feed)[0][0].astype(np.float64)


def ridge(x, y):
    """Return affine coefficients in input units, leaving the intercept unpenalized."""
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-10] = 1
    z = (x - center) / scale
    weights = np.linalg.solve(z.T @ z + 8 * np.eye(z.shape[1]), z.T @ (y - y.mean())) / scale
    return np.r_[weights, y.mean() - center @ weights]


def apply(x, affine):
    return x @ affine[:-1] + affine[-1]


def folded(a, intercept, basis, center, affine):
    current = a[:48] @ basis
    bias = intercept - current @ center + a[64] * affine[-1]
    cross = np.diag(a[56:64].copy())
    for (j, k), coefficient in zip(PAIRS, affine[:-1], strict=True):
        cross[j, k] += a[64] * coefficient
    return current, bias, a[48:56], cross


def score_program(h, current, previous, parameters):
    w, bias, past, cross = parameters
    return h @ w + bias + previous @ past + ((current @ cross) * previous).sum(axis=-1)
