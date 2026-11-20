# What does the previous dialogue turn add?

The contextual transfer passed the registered precision and recall targets on previously unseen EmoContext final messages. The paired test for a sender advantage missed its criterion. It caught 62 of 239 angry messages and raised 22 false alarms on the reference CPU. The equally sized local model caught 64 with 24 false alarms. Each receiver used one cached eight dimensional state from the preceding turn. The original sentiment output was preserved on every official input on each CPU.

## Prospective evaluation

The final EmoContext split contains 5,509 conversations and 298 angry final messages. The ordered gold labels in the raw participant TSV agreed with the independent SemEvalWorkshop conversion for all 5,509 records. The primary test removed 490 final utterances present in prior sources and 644 further repeats or empty targets, retaining 4,375 conversations with 239 angry final messages (5.46%). This screens exact normalized test targets. It cannot identify overlap in backbone pretraining or determine whether multiple conversations share a user. The task label is the emotion of the last message. No EmoContext labels were used for fitting or selecting cutoffs. [Registration](https://github.com/ruvnet/LatentMesh/commit/2cb83f1f2e6e92ca6a42b3e3525a7f9e413f775b) · [Weights and cutoffs published before inference](https://github.com/ruvnet/LatentMesh/commit/4e17176b7245f7b5566c0914a665208cdfb75bb1) · [Original task](https://aclanthology.org/S19-2005/).

### Registered primary cutoff on previously unseen text

| Receiver | True tags | False tags | Precision | Recall | F0.5 | ruOS precision / recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contextual transfer | 62 | 22 | 73.81% | 25.94% | 53.91 | 73.49% / 25.52% |
| Contextual local | 64 | 24 | 72.73% | 26.78% | 54.15 | 72.73% / 26.78% |
| Current message transfer | 34 | 7 | 82.93% | 14.23% | 42.18 | 82.50% / 13.81% |
| Current message local | 33 | 9 | 78.57% | 13.81% | 40.54 | 78.57% / 13.81% |
| Same transfer with zero state | 50 | 15 | 76.92% | 20.92% | 50.10 | 76.92% / 20.92% |

For contextual transfer the separate 95% intervals were +64.10 to +83.12% in precision and +20.51 to +31.70% in recall. The contextual transfer minus contextual local difference in F0.5 was **−0.23 points**, paired 95% interval **−2.44 to +1.88**. On ruOS it was −0.73, interval −3.26 to +1.56. The difference versus the locked current message transfer was +11.73 points, interval +4.34 to +19.80. Holding the contextual weights fixed and changing only the supplied previous state gave +3.81 points versus a reset, interval −2.21 to +10.34. Paired intervals use 10,000 resamples of whole three turn conversations. Each contributes one final prediction. Comparing two separately fitted readouts also changes the features and selected cutoffs. Only the state reset comparison holds all fitted weights and thresholds fixed.

### Preset F0.5 cutoff on previously unseen text

| Receiver | True tags | False tags | Precision | Recall | F0.5 | ruOS precision / recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contextual transfer | 25 | 8 | 75.76% | 10.46% | 33.69 | 75.76% / 10.46% |
| Contextual local | 24 | 9 | 72.73% | 10.04% | 32.35 | 72.73% / 10.04% |
| Current message transfer | 49 | 18 | 73.13% | 20.50% | 48.32 | 73.13% / 20.50% |
| Current message local | 58 | 23 | 71.60% | 24.27% | 51.51 | 71.60% / 24.27% |
| Same transfer with zero state | 18 | 5 | 78.26% | 7.53% | 27.19 | 78.26% / 7.53% |

At this secondary cutoff, the contextual transfer minus current message transfer F0.5 difference was −14.63 points. The current message transfer itself achieved 73.13% precision and 20.50% recall on the primary population. The choice of operating point matters when attributing gains to previous messages.

### Complete official test and published benchmark

| Receiver | True tags | False tags | Precision | Recall | F0.5 | ruOS precision / recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contextual transfer | 70 | 27 | 72.16% | 23.49% | 51.02 | 71.88% / 23.15% |
| Contextual local | 71 | 29 | 71.00% | 23.83% | 50.86 | 71.00% / 23.83% |
| Current message transfer | 37 | 10 | 78.72% | 12.42% | 38.07 | 78.26% / 12.08% |
| Current message local | 36 | 12 | 75.00% | 12.08% | 36.73 | 75.00% / 12.08% |
| Same transfer with zero state | 53 | 19 | 73.61% | 17.79% | 45.22 | 73.61% / 17.79% |

On the complete published split, contextual transfer had 35.44% anger F1 at the registered primary cutoff and 17.21% at the preset F0.5 cutoff. The 2019 shared task winner reported 80.58% anger F1, 77.23% precision and 84.23% recall on the same official test. SemEval provided task training labels and two preceding turns. Our receiver used zero EmoContext training labels and one preceding turn. The original overall leaderboard ranked multiclass micro F1. [Organizers' per class results, Table 5](https://aclanthology.org/S19-2005.pdf).

## What the sender supplied

Training and validation were identical to the prior [DailyDialog readouts](../dailydialog/README.md). We selected 768 training labels from a labelled pool of 69,566 and used 5,617 separate validation labels. Both models used the same 48 current principal coordinates, eight standardized previous coordinates and eight products. The final input was either the transferred 13 float score or the earlier no message score. The 65 input WSP solve used the pinned ruv-FANN implementation and no gradient updates. At boundaries the previous state is zero. A prediction on a new turn sends no data to a teacher or endpoint. Previous work measured the sender's initial 52 byte payload inside a 218 byte signed frame on provisioned hosts.

A separate label free diagnostic fitted an ordinary least squares reconstruction of the sender score on the 768 training examples. On the 5,617 validation examples, receiver features alone explained **98.60%** of the score variance. Including the local score explained **98.76%**. This measures predictability of the transferred scalar in DailyDialog. It does not measure anger accuracy or rule out useful information in the residual. [Diagnostic and exact values](reference/redundancy.json).

The current sender supplies a fixed affine score of the receiver's current hidden state. The matched local receiver crosses the same precision and recall thresholds in this study. Prefer it as the baseline for an optional tag. For the next signed packet, select a feature of the sender's dialogue context that improves anger prediction beyond the local features. Freeze that direction and the local control before scoring a fresh corpus. Keep the same label budget, a single 218 byte frame, exact native logits, at least 50% precision and 20% recall, and require the paired 95% lower bound on incremental F0.5 to exceed zero.

## Stateful installation and measured cost

The ONNX installation adds 12 operators and 56,672 serialized bytes to the pinned quantized RoBERTa. The original 1,551 operators and 355 tensors preserve their original serialized bytes. Both machines independently constructed the identical contextual transfer graph `6747e62092f2ed4e00fcbc46366ce6f14e6b9a82cdf7a5e0985786028e36c044`. Its state is eight doubles, 64 bytes per live conversation. No prior activation vector or extra encoder pass is needed in a running conversation. With isolated pairs, each preceding message must be encoded once to initialize its state.

The unmodified base, the instrumented model and both installations agreed exactly on the native logits of all 5,509 official final messages on each host. The produced state was checked against an independent computation for preceding and final messages. The installed primary and secondary booleans were checked against a separate NumPy calculation of the registered readouts. Across CPUs, 1 primary contextual transfer decisions and 0 primary contextual local decisions differed. Native preservation compares each installation to its own host. This dataset has no separate sentiment gold. The batch smoke test resets one dialogue while continuing another, checks identical native logits, and rejects one shared explicit state for a two row batch.

| Warm per turn inference | Native p50 | Context p50 | Native p95 | Context p95 |
| --- | ---: | ---: | ---: | ---: |
| Reference CPU | 7.71 ms | 8.04 ms | 10.49 ms | 10.42 ms |
| ruOS CPU | 11.72 ms | 11.41 ms | 16.44 ms | 16.00 ms |

These measurements use pretokenized final turns, 8 warmups and 128 measured pairs, with order rotated. Timing is sensitive to host load. [Reference scorecard](reference/decision.json) · [Independent ruOS scorecard](ruos/decision.json) · [Reference timing](reference/latency.json) · [ruOS timing](ruos/latency.json).

## Use and reproduce

Start with the [pinned base ONNX](https://huggingface.co/Xenova/twitter-roberta-base-sentiment-latest/blob/f3ec4d0925f90c3ca7ee7814f52d6ee7cf180445/onnx/model_quantized.onnx) and the [harness dependencies](../../../README.md). Run from the LatentMesh checkout.

```sh
python -m pip install -r harness/wsp/language/requirements.txt duckdb==1.4.4
f=harness/wsp/frontier/confidence/context
python "$f/receiver.py" target-model.onnx receiver.onnx --arm transfer
OPENBLAS_NUM_THREADS=1 python "$f/score.py" verify "$f/reference"
OPENBLAS_NUM_THREADS=1 python "$f/score.py" verify "$f/ruos"
OPENBLAS_NUM_THREADS=1 python "$f/score.py" compare "$f/reference" "$f/ruos"
bash "$f/replay.sh" ./model-cache ./fresh-context-run --refit
OPENBLAS_NUM_THREADS=1 python "$f/redundancy.py" ./fresh-context-run/features.json ./fresh-context-run/redundancy.json
```

The optional refit writes its projection and weights to the new run directory. Inference always uses the registered lock. Select `--arm local` to install the matched control. Load the wrapper from the context directory. A new conversation may omit the prior state or supply an explicit zero row.

```python
from stream import step

_, _, _, _, state = step(session, previous_turn_tokens)
sentiment_logits, score, anger, anger_f05, state = step(session, current_turn_tokens, state)
```

Keep each returned state with its own conversation. Dropping it starts a new conversation. Select the original checkpoint to roll back.
