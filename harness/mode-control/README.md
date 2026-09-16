# LatentMesh reasoning-mode control (experimental)

Small, explicit **mode advisories**, not transferred thoughts. Zero dependencies;
Node >=20. Run `npm test` in this directory.

`lib/control.mjs` exports:

- `detectMode(activation, probe)`: locally supplied sparse linear classifier with
  standardized feature support checks, stable softmax and abstention.
- `signMessage(payload, privateKey)` and `ModeReceiver`: Ed25519 envelope, pinned
  target revision/adapter, expiry and session-specific replay rejection.
- `loadAdapter(bytes, expectedHash)`: verified, deeply frozen local directions.
- `steerResidual(residual, options)`: copy-on-write bounded additive experiment,
  observe-only by default, with deterministic before/after hash receipts.

The end-to-end test demonstrates probe → signature → receiver → local operator.
Its tiny vectors are **synthetic correctness fixtures**, never research evidence.
Use the exact schemas in that test as the API example. Persist the receiver's
`lastSequence` across restart; serialize receive calls per sender/session.

Mode transport and activation use must remain behind the existing authority and
causal-admission layers. No production router or Candle hooks are installed.
No calibrated probes, trained directions, model accuracy or speedup are claimed.
See [ADR-049](../../docs/adr/049-reasoning-mode-control.md) for controlled evaluation
and the remaining live-runtime work.
