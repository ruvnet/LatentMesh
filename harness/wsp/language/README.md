# Transferring anger recognition between two pretrained language encoders

DistilBERT supplied a supervised direction from its hidden state to an intact RoBERTa sentiment model. The receiver added a 12 weight, one bias output to recognize anger among four other negative emotions. The hybrid scored **67.94%** on 262 reserved examples. A compact text encoding of the same hidden evidence scored **69.47%**. The additional response also displaced 11 previously correct sentiment decisions among 184 separate examples.

| Receiver signal | Anchors | Useful payload bytes | Anger task | Retained sentiment |
| --- | ---: | ---: | ---: | ---: |
| Intact sentiment receiver | 0 | 0 | 50.00% | 85.87% |
| Hidden direction, 8 bit | 48 | 52 | 67.94% | 79.89% |
| One bit teacher | 48 | 6 | 66.41% | 82.61% |
| One bit teacher | 64 | 8 | 68.32% | 82.61% |
| Decimal confidence as text | 52 | 52 | 68.70% | 79.35% |
| Four bit confidence as base64 text | 64 | 44 | 69.47% | 79.89% |
| Short human readable phrase | 64 receiver comparisons | 49 | 54.58% | 26.63% |
| Native sender sentiment | 48 | 52 | 50.38% | 9.78% |
| Shuffled hidden values | 48 | 52 | 50.38% | 37.50% |

**Decision.** Keep this adapter in the research harness. The 8 bit transfer trailed the strongest comparator by 1.53 percentage points. Its paired 95% bootstrap interval ranged from 4.20 points behind to 1.15 ahead. Retained task accuracy fell 5.98 points, with a paired interval from 2.72 to 9.78 points lost. The registered thresholds were a 2 point gain and at most a 1 point loss. A new experiment could calibrate abstention on a separate development population before testing fresh examples.

**What transferred.** The sender's hidden anger direction ranked positives above negatives with AUROC 0.770. Its native negative sentiment probability scored 0.482. This is evidence for a useful hidden feature on this task. The best comparator sent 64 coarser observations of that same feature as printable characters. The literal phrase was a separate control. Relative precision and number of distinct anchors both varied in the fixed budget comparison.

**What changed.** The original transformer parameters stayed frozen. The experiment attached a competing readout to a receiver specific PCA projection. The sender standardized its own hidden representation and fitted a supervised direction with disjoint labeled training examples. The receiver fitted its adapter by closed form WSP without gradients or receiver probe labels. Both sides receive a deliberately balanced public anchor bank. A recipient with independent access to the original dataset could look up anchor annotations. This benchmark does not estimate task prevalence or deployment accuracy.

**Transport and provisioning.** Each transmitted arm used 52 available payload bytes, 154 signed application bytes and one 218 byte C Air loopback frame. Both ends already had the ordered anchor bank (5,032 bytes with IDs, 4,071 bytes of text), an arm manifest (1,740 to 1,744 bytes) and a trusted public key (44 bytes in SPKI). The fitted receiver basis and centering vector occupy 79,872 bytes in float64. The source direction and scaling occupy 18,440 bytes locally. The pinned ONNX graphs occupy 67,581,197 and 125,905,426 bytes before tokenizers. The 218 byte figure represents warm delivery over the host codec. Physical RF latency, loss and end device integration remain unmeasured.

## Independent replay

A second CPU on ruOS downloaded the same pinned weights and data, fitted fresh parameters and scored the same 446 cases independently. Its hybrid scored **67.56%** on anger and **80.43%** on retention. Its best two controls tied at **69.47%**. It missed the registered thresholds by the same decision rule. One hybrid prediction in each population differed across hosts. The receiver basis and sender direction fingerprints also differed. Both hosts verified 13 edits per arm and identical 218 byte frames. The exact source of the numerical variation was not isolated. The published reference and ruOS receipts preserve both results; neither was substituted for the other.

With cached receiver features the median signature verification and patch solve took 0.69 ms on the reference host and 1.02 ms on ruOS. The corresponding p95 times were 1.11 and 3.69 ms. Median tokenization and forward inference per text took 5.93 ms for the sender and 10.97 ms for the receiver on the reference host, or 13.12 and 25.31 ms on ruOS. These are local CPU observations with batch size one and two inference threads. [ruOS scorecard](ruos/decision.json) · [Paired ruOS observations](ruos/predictions.json) · [Timing samples summary](timings.json)

## Reproduce

Use the sibling repository checkout and companion revision in the [parent harness](../README.md). Also install Python 3.12, Node 20.9 or newer, and a C11 compiler. The replay fetches SHA256 verified public exports and data into the first directory. The second directory holds arrays, adapter weights, predictions and host timings. Original text and model weights are excluded from Git.

```sh
python3.12 -m venv /tmp/language-wsp-venv
. /tmp/language-wsp-venv/bin/activate
python -m pip install -r harness/wsp/language/requirements.txt
bash harness/wsp/language/replay.sh /tmp/language-cache /tmp/language-run
```

To verify the published receipts, thresholds and paired intervals without downloading the checkpoints, install NumPy 2.3.5 and run this from the LatentMesh checkout.

```sh
python harness/wsp/language/check.py harness/wsp/language harness/wsp/language --exact
python harness/wsp/language/check.py harness/wsp/language harness/wsp/language/ruos
```

The [first protocol](https://github.com/ruvnet/LatentMesh/blob/8800dd1d69a7ff09717b3c0533de91fde8dc4c8b/harness/wsp/language/protocol.md) precedes the [corrected probe order](https://github.com/ruvnet/LatentMesh/commit/1222711d1311c997a0e16ade673aa19341dea22a). The [frozen candidate fingerprints](https://github.com/ruvnet/LatentMesh/blob/7ad402c2252f7c9e668fe971989cd000313ace68/harness/wsp/language/lock.json) and scoring code were committed before extracting held out data. [Complete scorecard](decision.json) · [Paired predictions and actual costs](predictions.json) · [GoEmotions dataset paper](https://aclanthology.org/2020.acl-main.372/) · [Source checkpoint](https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english/tree/0b6928efcb76139cae2c6881d49cda67fe119f42) · [Receiver checkpoint](https://huggingface.co/Xenova/twitter-roberta-base-sentiment-latest/tree/f3ec4d0925f90c3ca7ee7814f52d6ee7cf180445)
