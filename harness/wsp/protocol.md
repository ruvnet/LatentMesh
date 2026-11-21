# Controlled readout repair

Freeze the following comparison for the prototype. The experiment is registered
in the project coordination issue before scoring. All ten runs are reported.

1. Use the pinned UCI Iris and Wine data, the upstream five stratified splits
   (17, 43, 101, 211, 509), and training only standardization. Deduplicate exact
   feature vectors in training, probe, then test order. Record both split hashes.
2. Fit two independent classifiers on the same training examples using the
   existing scalar WASM trainer, AdamW, 120 epochs, learning rate 0.01. The
   sender has one hidden Tanh layer of width 8. The receiver has two Tanh layers
   of widths 7 and 5 and a different initialization. Set row 2 and its bias to
   zero in a copy of each model. The intact sender provides the correction.
3. Use validation features as shared, ordered probes. The content of the q8
   message is the change in a single sender output on those probes. The decoder
   cannot inspect sender weights. Fit once using the receiver's hidden features.
   This is output distillation of a repair into an already learned basis.
4. Compare the untouched damaged receiver, its intact reference, full precision
   output changes, q8 changes, a packed one bit teacher membership per probe,
   shuffled q8 association, and seeded random values with the same L2 norm.
   The two deployable teacher arms use the same signing, transport and solver.
   Training labels or test labels are never supplied to either patch fitter.
5. Score on the reserved test examples. Report accuracy, macro F1, MSE,
   retained class accuracy, sibling output drift, quantization error, edits and
   frame bytes. Report each dataset's mean across splits. The same examples can
   recur in different seeds, so a pooled sample count is not an independent
   population size. Wire bytes exclude provisioning, firmware and RF overhead.
6. To justify a larger trial, require q8 accuracy to exceed the one bit teacher
   by at least 2 percentage points on **each** dataset, retain original class
   accuracy within 1 point of the intact receiver on each, and preserve every
   sibling output exactly. A null result is a valid run. Any runtime admission
   would require a new model pair, fresh task and the existing causal controls
   including text and self generated content.
