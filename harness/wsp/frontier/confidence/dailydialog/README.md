# Does the transferred anger tag travel to DailyDialog?

On 5,072 previously unseen turns from 844 DailyDialog conversations, the frozen MELD tag missed the registered requirement of 50% precision, 20% recall and identical native sentiment logits. It found 12 of 67 angry lines and emitted 86 false tags. With DailyDialog supervision the adapted transfer found 9 angry lines and emitted 19 false tags. The equally sized local WSP control found 8 and emitted 14 false tags.

## Previously unseen utterances

DailyDialog is an independently collected set of human written conversations. Of 7,740 official test turns, 2,643 matched earlier data after normalization and 25 more repeated or normalized to empty. The primary population retained 5,072 turns, including 67 labelled angry (1.32%). The prior MELD population had 7.51% anger. Population and annotation differences both affect precision. Our encoder saw individual utterances. Exact text filtering cannot rule out near duplicates or overlap with backbone pretraining. [Original data and paper](https://aclanthology.org/I17-1099/).

| Receiver | True tags | False tags | Precision | Recall | F0.5 | ruOS precision / recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen 13 float WSP, score alone | 12 | 86 | 12.24% | 17.91% | 13.07 | 12.12% / 17.91% |
| Frozen 13 floats with sentiment confidence | 9 | 53 | 14.52% | 13.43% | 14.29 | 14.52% / 13.43% |
| Frozen local receiver | 19 | 253 | 6.99% | 28.36% | 8.23 | 6.99% / 28.36% |
| Frozen 48 coefficient WSP | 25 | 316 | 7.33% | 37.31% | 8.74 | 7.65% / 38.81% |
| Frozen soft distillation | 18 | 263 | 6.41% | 26.87% | 7.56 | 6.83% / 28.36% |
| Adapted PCA plus transferred score | 9 | 19 | 32.14% | 13.43% | 25.14 | 34.62% / 13.43% |
| Adapted PCA plus local score | 8 | 14 | 36.36% | 11.94% | 25.81 | 36.36% / 11.94% |
| Adapted PCA alone | 8 | 17 | 32.00% | 11.94% | 23.95 | 33.33% / 11.94% |

The frozen transfer precision interval was 5.05 to 20.18%, and its recall interval was 8.77 to 26.98%. The adapted transfer missed the point gate on the reference CPU, with precision interval 10.00 to 52.38% and recall interval 3.64 to 25.40%. Point estimates and 95% intervals answer different questions.

The adapted transfer minus the equally sized local receiver was **−0.67 points in F0.5**, paired 95% interval **-5.17 to 6.27**. On ruOS the difference was +0.51, interval -3.54 to 7.35. The study required a positive paired lower bound to credit the sender. Both CPUs scored the same turns in two independent executions. Each interval resamples whole dialogues 10,000 times with the same draws for all methods.

## Threshold sensitivity and complete official test

Both operating points were frozen before test access. The primary adapted threshold maximized validation recall subject to 50% precision and at least 20 tags. The secondary threshold maximized F0.5. Its reference results on previously unseen turns follow.

| Receiver | True tags | False tags | Precision | Recall | F0.5 | ruOS precision / recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adapted transfer, F0.5 threshold | 13 | 34 | 27.66% | 19.40% | 25.49 | 27.66% / 19.40% |
| Adapted local, F0.5 threshold | 12 | 37 | 24.49% | 17.91% | 22.81 | 24.49% / 17.91% |
| Adapted PCA alone, F0.5 threshold | 13 | 39 | 25.00% | 19.40% | 23.64 | 25.00% / 19.40% |

On the complete original test there were 7,740 turns in 1,000 conversations and 118 angry lines. This secondary population includes turns also seen in development.

| Receiver | True tags | False tags | Precision | Recall | F0.5 | ruOS precision / recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen 13 float WSP, score alone | 32 | 133 | 19.39% | 27.12% | 20.57 | 19.39% / 27.12% |
| Adapted PCA plus transferred score | 20 | 33 | 37.74% | 16.95% | 30.30 | 38.00% / 16.10% |
| Adapted PCA plus local score | 15 | 24 | 38.46% | 12.71% | 27.37 | 37.50% / 12.71% |
| Adapted PCA alone | 19 | 32 | 37.25% | 16.10% | 29.50 | 36.73% / 15.25% |

These are binary anger tag metrics. Published multiclass DailyDialog evaluations also report micro F1 excluding neutral and macro F1 across emotion classes. Leaderboard comparison requires matching tasks and metrics. [Published benchmark description](https://aclanthology.org/2021.findings-emnlp.104/).

## Can a different cutoff reach the gate?

After recording the reference result, we checked every possible single cutoff on each locked score using the same clean test labels. This gives an upper bound on this sample rather than a deployable setting. At least 14 of the 67 angry turns must be recovered to reach 20% recall. Under that constraint the frozen transfer topped out at **11.48% precision**, the adapted transfer at **23.16%**, and the local control at **23.33%**. At 50% precision the adapted transfer could recover at most 8.96% of angry lines. Its average precision rose from 9.65% for the frozen score to 18.00%, close to 17.83% for local supervision. The score needs to rank angry examples higher for another cutoff to achieve the gate. A prospective follow up could add the previous turn as an input and compare with the same local model and label budget on separate conversations. [Diagnostic design](ranking.md) · [Rank counts and exact arithmetic](reference/ranking.json).

## Where the weights and labels come from

The zero target label comparison reused the [frozen 13 float transfer](https://github.com/ruvnet/LatentMesh/commit/9136c188ec18d4bad7698cb3e0414b45d2ad2de5) and the MELD calibration lock. The previous sender measurement was 52 useful bytes and one 218 byte signed C loopback frame between provisioned endpoints. DailyDialog calibration requires zero further sender bytes.

The new readouts received 768 labelled training turns (192 angry) from 695 conversations, chosen by class from a labelled training pool of 69,566. Threshold selection used all 5,617 eligible validation turns, including 53 angry, from 884 conversations. One training dialogue had 12 text turns and 11 labels. It was excluded in full. The identical budget applies to all adapted controls.

We computed 64 principal directions from the receiver's training states, then used ruv-FANN to fit one linear weight edit. The transfer and local controls each had 65 inputs and a bias. Their only difference was the last input, either the sender's 13 float score or the previous local readout. A third control used the 64 receiver features alone. We folded the learned projection and score into one 768 coefficient weight vector and a bias. Production inference needs a single added matrix multiplication. [Registration](https://github.com/ruvnet/LatentMesh/commit/ed90ee5eb3ef564b129a36e37009eeb670477211) · [Frozen coefficients](https://github.com/ruvnet/LatentMesh/commit/ee71f9dc83ea73aca512f8979690d23faa55bdfb) · [Reference scorecard](reference/decision.json) · [ruOS scorecard](ruos/decision.json).

## Install and replay

Start with the [pinned original RoBERTa checkpoint](https://huggingface.co/Xenova/twitter-roberta-base-sentiment-latest/blob/f3ec4d0925f90c3ca7ee7814f52d6ee7cf180445/onnx/model_quantized.onnx). Both machines rebuilt the same transfer graph, SHA256 `d39c71b00321fbddfb8afebe6bb61076d694a3103b48c654d5bc58ad2d3d3e5c`. The installed graph adds 6,649 serialized bytes and 5 pure operations. All 1,551 original operators and 355 parameter tensors are preserved. On all 7,740 official test inputs, the two installations were checked against the unmodified base. See each [reference](reference/graph.json) and [ruOS](ruos/graph.json) receipt and each scorecard's parity measurements. An additional [128 turn installation check](reference/controls.json) verified the transferred, matched local and PCA only graphs, including both output thresholds. [Independent control builds](ruos/controls.json). Each installed graph had exactly zero native logit drift on all 7,740 official turns relative to its own unedited CPU baseline. Across CPUs, two frozen tag decisions and three primary adapted tag decisions differed on the full test. DailyDialog supplies emotion labels. The sentiment preservation measurement uses logit equivalence because this dataset lacks separate three way sentiment gold.

```sh
python -m pip install -r harness/wsp/language/requirements.txt
python harness/wsp/frontier/confidence/dailydialog/install_receiver.py target-model.onnx receiver.onnx --arm transfer
```

Use `--arm local` for the equally sized no message control or `--arm pca` for the 64 feature control. The three outputs in addition to the base's original `logits` are `dd_score`, `dd_anger` (the primary threshold) and `dd_anger_f05` (the secondary threshold). Applications can consume the extra boolean alongside the original sentiment output. Rollback selects the unmodified checkpoint.

```python
sentiment_logits, anger = session.run(["logits", "dd_anger"], tokenized_inputs)
```

To replay raw inference, use the pinned sibling ruv-FANN checkout in the [harness setup](../../../README.md), and a new run directory. Add `--refit` to independently fit the coefficients on that host before evaluating the committed graph. The compact replay only needs NumPy and the committed observations.

```sh
python -m pip install duckdb==1.4.4
f=harness/wsp/frontier/confidence/dailydialog
bash "$f/replay.sh" ./model-cache ./daily-run
python "$f/ranking.py" run ./model-cache ./daily-run
OPENBLAS_NUM_THREADS=1 python "$f/report.py" verify "$f/reference"
OPENBLAS_NUM_THREADS=1 python "$f/report.py" verify "$f/ruos"
OPENBLAS_NUM_THREADS=1 python "$f/report.py" compare "$f/reference" "$f/ruos"
python "$f/ranking.py" verify "$f/reference"
```
