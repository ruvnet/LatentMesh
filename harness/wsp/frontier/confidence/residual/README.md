# Does a residual weight program add transferable anger information?

Keep the local receiver as the default. On 2,011 previously unseen final messages, the signed program detected two more angry conversations than matched local with the same 12 false alerts. With only 12 disagreements, the paired F0.5 difference of +2.31 points has a 95% interval from −5.28 to +10.14. The shuffled sender also came within one detection of the real sender.

## Primary outcome

The [protocol](protocol.md) was committed at `bcf205e` before fitting. The [numerical lock](lock.json), operating points and [signed fixtures](packets.json) were committed at `43fdb8b` before scoring the independent EmoContext Test1 split. We verified all 2,755 gold labels against a second distribution. Removing 718 targets present in historical corpora or any previous Test2 turn and 26 repeated or empty targets left 2,011 conversations, including 106 angry final messages. Test1 and the earlier Test2 come from the same benchmark. This experiment evaluates binary anger detection using the final two turns. [Official task description](https://aclanthology.org/S19-2005/)

| Primary program on screened Test1 | True alerts | False alerts | Precision | Recall | F0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Residual sender packet | 27 | 12 | 69.23% | 25.47% | 51.53% |
| Matched local fit | 25 | 12 | 67.57% | 23.58% | 49.21% |
| Shuffled sender packet | 26 | 12 | 68.42% | 24.53% | 50.39% |
| Unrestricted per example sender | 20 | 9 | 68.97% | 18.87% | 45.05% |
| Prior contextual local | 26 | 13 | 66.67% | 24.53% | 49.62% |
| Prior contextual packet | 26 | 13 | 66.67% | 24.53% | 49.62% |

The registered minimums were 50% precision, 20% recall, identical native logits and a strictly positive paired 95% lower bound against matched local. Real versus shuffled was +1.14 F0.5 points (−3.27 to +5.87) across five disagreements. Against the prior local program the gain was +1.91 points (0.00 to +5.12). On all 2,755 official cases the real program scored 71.43% precision and 20.00% recall, with a paired gain of +4.12 points over matched local (−2.95 to +11.58). Both operating points and both populations are in the [reference scorecard](reference/decision.json). Intervals resample whole paired conversations 10,000 times. Repeating inference on another CPU tests numerical reproducibility on the same cases.

## Where the information was lost

All four newly fitted heads used the identical 768 DailyDialog training examples, including 192 positives, the same 5,617 validation turns, including 53 positives, and the same closed form ruv-FANN WSP solver. Training selected the 768 examples by screening 69,566 labeled eligible training turns. The 416 public enrollment pairs each came from a different training dialogue outside the fitting groups. Enrollment used their texts and public model activations. The real and shuffled packets differ only in whether sender scores preserve their association with the anchor text. Local directly fits the same twelve products to the same out of fold targets. No candidate used a Test1 label to fit a coefficient or select a cutoff.

The full DistilBERT sender tried to predict the errors the contextual RoBERTa receiver made on held out training dialogues. On separate DailyDialog validation turns the sender's correlation with residual error was −0.067. The compiled packet's correlation with the full sender score was 0.0013 (R² −0.036). At the primary cutoff on screened Test1, giving the final readout the online sender score yielded −4.17 F0.5 points versus matched local (−13.69 to +5.25). The full sender is a diagnostic and carries no claim to the wire budget. A stronger source must first demonstrate conditional value before a new codec or larger packet can be justified. [Validation diagnostic](reference/diagnostic.json)

## A second sender, screened before compression

On the previously used DailyDialog validation split we tested the unrestricted native anger margin from a separate Twitter emotion checkpoint. Both linear receivers received the same 768 fitting labels. Adding the sender lowered average precision from 42.89% to 42.38% on 5,617 validation turns from 884 conversations, including 53 angry turns. The paired difference was −0.51 points (95% dialogue bootstrap interval −2.28 to +1.15). Its partial correlation with the local receiver error was −0.039. This sender missed the development screen, so the experiment ended before fitting another packet. Both checkpoints in this follow up belong to the RoBERTa family. The independent Test1 trial above used DistilBERT as the sender. [Development protocol](preflight.md) · [Recorded score](reference/preflight.json)

A subsequent prospective screen tested a full EmoBERTa dialogue sender on 4,096 new conversations. The paired gain over matched local was +1.41 F0.5 points, with a 95% interval from −1.17 to +3.93. The remaining 4,096 conversations are reserved. [Dialogue sender experiment](../dialogue/README.md)

## Deployment cost

The receiver stores eight doubles (64 bytes) from the preceding turn. It uses the existing 768 dimensional hidden state, eight diagonal interactions and twelve additional off diagonal interactions. The new optional ONNX graph adds 13 nodes and 57,281 bytes to the original 125,905,426 byte receiver. The original 1,551 nodes and 355 initializers remain byte identical. The sender model is 67,581,197 bytes and is needed only during enrollment or for the unrestricted diagnostic. On provisioned hosts each matched real or shuffled packet costs 52 useful bytes, 154 signed application bytes and one 218 byte C Air frame. Initial distribution of the models, projection, public anchor data and local WSP readout fitting is separate.

Both CPUs agreed on every binary decision across all six methods, both cutoffs and all 2,755 conversations. Every installed program matched its host's original sentiment logits bit for bit. The original checkpoint produced different raw logit digests across the two hosts. [Independent run](ruos/decision.json)

| CPU and warmed latency | Native | Residual sender | Matched local |
| --- | ---: | ---: | ---: |
| Reference p50 | 8.38 ms | 8.32 ms | 8.75 ms |
| Reference p95 | 16.46 ms | 12.00 ms | 13.79 ms |
| ruOS p50 | 13.83 ms | 13.75 ms | 14.88 ms |
| ruOS p95 | 21.63 ms | 22.90 ms | 25.15 ms |

The timing sample used the first 128 pretokenized Test1 final turns after eight warmups, two inference threads and rotated evaluation order. Past state was already available. Small timing differences are descriptive. The two CPUs are independent execution environments. [Reference timing](reference/latency.json) · [ruOS timing](ruos/latency.json)

## Reproduce

From the LatentMesh checkout, replay the stored observations with NumPy 2.3.5, Node and a C compiler. The source checkout for training is ruvnet/ruv-FANN at `1abb28d3d23f00c04810f59ff7b3797b57bdc24c`.

```sh
r=harness/wsp/frontier/confidence/residual
node "$r/wire.mjs" verify "$r"
OPENBLAS_NUM_THREADS=1 python "$r/score_results.py" compare "$r/reference" "$r/ruos"
```

For live inference or enrollment install the pinned harness dependencies, including DuckDB for reading the historical overlap lists. The optional program can be installed on a copy of the pinned receiver. The file destination must be new. Omit previous state at each conversation start. The original sentiment logits are available alongside the optional tags. [Streaming input and output contract](../context/stream.py)

```sh
python -m pip install -r harness/wsp/language/requirements.txt duckdb==1.4.4
python "$r/runtime.py" target-model.onnx edited.onnx --arm transfer
bash "$r/replay.sh" ./model-cache ./new-test1
python "$r/timing.py" ./model-cache ./new-test1/latency.json
```

To rebuild the candidate on the reference stack, regenerate or reuse the prior contextual development cache. Its ordered features and projection are checked against the previous lock. Signing uses a fresh ephemeral key for each enrollment. Compare the decoded programs and numerical parameters when refitting.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python "$r/../context/develop.py" prepare ./model-cache ./context-fit
python "$r/train.py" enroll ./model-cache ./context-fit ./residual-fit
node "$r/wire.mjs" send ./residual-fit
python "$r/train.py" features ./residual-fit
node "$r/../dailydialog/fit.mjs" ./residual-fit --residual
python "$r/train.py" freeze ./context-fit ./residual-fit
node "$r/wire.mjs" verify ./residual-fit
python "$r/diagnostic.py" ./context-fit ./residual-fit
```
