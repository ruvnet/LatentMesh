# Can the receiver learn when to tag anger?

Training on anger against every other emotion improved the precision of the 13 float WSP transfer. On 2,277 utterances from 409 held out MELD dialogues, precision rose from 17.83% to 28.87%. Recall fell from 50.88% to 23.98%. The separate anger tag preserves the receiver's native three way sentiment output. Use it as an experimental signal. The registered automatic override failed its adoption test.

## An anger tag in complete dialogues

The receiver saw only the utterance text. Every calibrated method got the same 768 fit labels from 273 dialogues and the same 3,224 threshold selection labels from 308 other dialogues. The final 409 dialogues were disjoint from both groups and excluded normalized text found in the earlier evaluations. Of the 2,277 test utterances, 171 were angry (7.51%). The old dialogue calibrated tag and all new policies were evaluated on exactly these rows. The held out dialogues are from the same TV series. The two CPUs are independent executions on the same examples.

| Reference CPU, held out | Precision | Recall | F0.5 | True tags | False tags |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous dialogue calibrated 13 float WSP | 17.83% | 50.88% | 20.49 | 87 | 401 |
| 13 float WSP with native confidence | 28.87% | 23.98% | 27.74 | 41 | 101 |
| 13 float WSP using the transferred score alone | 28.99% | 28.65% | 28.93 | 49 | 120 |
| 48 coefficient WSP with native confidence | 20.06% | 42.11% | 22.40 | 72 | 287 |
| Conventional soft distillation | 20.34% | 35.09% | 22.21 | 60 | 235 |
| Local receiver with no message | 25.53% | 28.07% | 26.00 | 48 | 140 |
| Native confidence alone | 11.04% | 21.64% | 12.24 | 37 | 298 |

The preregistered 13 float confidence edit gained **11.05 percentage points in precision** over the previous receiver, paired 95% interval **5.40 to 17.02**. The F0.5 gain was 7.25 points, interval 2.03 to 12.53. The ruOS CPU measured a precision gain of 11.58 points, interval 5.87 to 17.58. All intervals resample whole dialogues 10,000 times. On the reference host the precision difference from the no message receiver was 3.34 points, interval −2.00 to 9.03. Transfer superiority to that control remains open.

Adding native confidence provided no measurable advantage over using the transferred score alone. The precision difference was −0.12 points, interval −3.49 to 3.13. The simpler tag produced the same 49 true and 120 false tags on both hosts. The more conservative 13 float confidence threshold had 50.75% precision on selection (34 of 67) but reached only 32.50% precision and 7.60% recall on the reference test. No registered policy reached both 50% precision and 20% recall on this population.

The native classifier alone was correct on 1,440 of 2,277 reference examples. Consuming the additional tag preserves these answers. Treating it as a replacement would discard 70 of them with native confidence and 87 with the score alone. These are measured consequences of the explicit route output, which applications can choose to ignore.

[Registration](precision/protocol.md) · [Coefficients frozen before testing](https://github.com/ruvnet/LatentMesh/commit/466a6611f6f7745e8b517e5d12894bdf06eaac7c) · [Reference record](precision/reference/decision.json) · [ruOS record](precision/ruos/decision.json)

## What the preceding binary test established

The sender's 13 float packet is itself WSP, representing a readout over twelve receiver coordinates and a bias. Its competitor uses 48 signed quantized coefficients. Seven transmitted programs used 52 payload bytes and one 218 byte signed frame on the C Air loopback, given the shared basis and fixed models. Every calibration experiment below reused the received program. Its extra receiver labels and local installation are separate costs.

| Reference CPU, anger versus sadness balanced accuracy | 13 float WSP | 48 coefficient WSP | Soft distillation |
| --- | ---: | ---: | ---: |
| MELD test, no dialogue calibration | 64.81% | 60.15% | 57.91% |
| MELD test, separate dialogue calibration | 66.12% | 65.23% | 60.62% |
| Fresh CARER validation, social media calibration | 63.65% | 61.55% | 62.66% |
| Previously unused MELD development dialogues, frozen tag | 67.84% | 58.87% | 58.59% |

The main MELD panel had 2,257 full utterances, including 321 angry and 193 sad. The protected automatic rule selected zero overrides and obtained 50% balanced accuracy. The criterion required a two point skill gain over the strongest control and at most one point of native sentiment loss. It failed. The optional tag was separately registered before test access.

The subsequent confirmation used 963 other utterances from 113 dialogues. On the 142 angry and 101 sad lines the 13 float receiver beat WSP48 by 8.98 points, paired 95% interval 2.17 to 14.59 on both CPUs. Across all 963 lines the reference tag found 128 angry utterances among 654 flags, only 19.57% precision. That discrepancy motivated the all emotion experiment above. These are text only binary and tagging tasks. A state of the art claim would require the same full benchmark, inputs and scoring as published peers.

[Initial protocol](protocol.md) · [Optional tag registration](separate-tag.md) · [Confirmation protocol](confirmation.md) · [Reference confirmation](reference/confirmation.json) · [ruOS confirmation](ruos/confirmation.json) · [MELD paper](https://aclanthology.org/P19-1050/)

## Run it

Use the exact quantized RoBERTa base specified in the [parent adapter instructions](../README.md). From the LatentMesh root, build the score only tag in a new destination file:

```sh
python -m pip install -r harness/wsp/language/requirements.txt
python harness/wsp/frontier/confidence/install.py target-model.onnx all f05 receiver.onnx --arm scoreOnly
```

```python
import onnxruntime as ort

session = ort.InferenceSession("receiver.onnx", providers=["CPUExecutionProvider"])
sentiment_logits, anger_tag = session.run(["logits", "wspc_tag"], tokenized_inputs)
```

Use `--arm float13` to include native confidence. The installer hashes the base and calibration lock and preserves the original 1,551 graph operations and 355 parameter tensors. The score only installation adds 6,800 serialized bytes beyond the preceding adapter, uses two local scalars, and needs no further sender traffic. On 64 held out reference inputs, the installed score only and confidence graphs produced identical native logits to the original and agreed with the independent tag decisions. CPU timings and file fingerprints are in the [graph receipts](precision/reference/graphs.json). Remove the experimental receiver from service by selecting the original base checkpoint.

For full inference, use the sibling ruv-FANN checkout pinned in the [harness setup](../../README.md), plus the Python dependencies above and DuckDB 1.4.4. A fresh run downloads pinned public checkpoints and datasets, independently refits and records local coefficients, then scores the precommitted programs. The compact replay reads archived predictions and does not need the checkpoints or raw dialogue.

```sh
python -m pip install duckdb==1.4.4
bash harness/wsp/frontier/confidence/replay.sh ./model-cache ./new-run

cd harness/wsp/frontier/confidence
OPENBLAS_NUM_THREADS=1 python precision/assess.py verify precision/reference precision/lock.json
OPENBLAS_NUM_THREADS=1 python precision/assess.py verify precision/ruos precision/lock.json
```

A deployment test should freeze the selected program and evaluate an independently sourced collection of whole dialogues at natural prevalence. Require at least 50% precision, at least 20% recall, and identical native sentiment logits. Compare against the no message receiver under the same label budget.
