# LatentMesh Air MetaHarness

This harness evolves radio policy while freezing the protocol, model, channel
suites, and promotion rule. It uses `@metaharness/flywheel` to produce signed,
replayable promotion evidence.

The current evaluator is a deterministic channel simulator. Its results are
labelled `simulated` and cannot satisfy the hardware acceptance gate in
`docs/air/ACCEPTANCE.md`. Hardware benchmark records can replace the evaluator
without changing the promotion contract.

The stage benchmark deliberately reports two ratios separately:

1. Semantic transport targets at least ten times fewer bytes with equivalent
   downstream task accuracy.
2. Neural PHY targets at least two times additional useful information per
   unit airtime or energy with the semantic message held fixed.

Neither simulated ratio is promoted into an over the air claim. The generated
receipt always records the missing hardware, confidence interval, and unseen
propagation evidence.

## Evolvable policy

* semantic payload budget
* FEC rate
* interleaver depth
* neural confidence threshold

## Frozen promotion rule

A candidate must improve task weighted useful information per second by at
least two percent, never reduce mean critical agreement, maintain at least 99
percent critical agreement on every holdout case, avoid higher fallback cost,
and remain within two percent of baseline on an unseen harder anchor suite.

```bash
npm install --ignore-scripts
npm run validate
```

The stage benchmark writes `artifacts/stage-benchmark.json`. The optimizer
writes its signed replay bundle and tuned policy to `artifacts/`.
