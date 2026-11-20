# Can a single frame teach RoBERTa a new response?

A DistilBERT sender compressed its anger signal on 416 shared texts into 48 quantized readout parameters. A provisioned RoBERTa receiver installed the readout through ruv-FANN without gradient descent. Seven transmitted methods each used one signed 218 byte C Air loopback frame. A 13 float readout was the most accurate compact method on both fresh binary tests, on both CPUs. The registered advancement decision is **hold**.

The [receiver calibration study](confidence/README.md) tests the 13 float and 48 coefficient programs on complete MELD dialogue populations, including every nonanger emotion. It provides independent CPU evidence and an installable additional ONNX output.

## Held out transfer

Anger accuracy is balanced accuracy on 512 cases per domain. Sentiment loss is the percentage of original test cases on which the intact receiver was correct and the new output displaced its answer. GoEmotions used a separate set of 700 joy and sadness examples for retention. TweetEval used 900 independently selected negative, neutral and positive sentiment examples.

| Host and domain | WSP anger | Soft distillation | 13 floats | WSP sentiment loss | Paired 95% interval, WSP minus 13 floats |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference, Reddit | 67.38% | 68.55% | 69.34% | 0.57 points (4/700) | −4.69 to +0.78 points |
| Reference, tweets | 56.84% | 57.23% | 59.77% | 6.67 points (60/900) | −5.08 to −0.98 points |
| ruOS, Reddit | 67.58% | 68.55% | 69.14% | 0.71 points (5/700) | −4.10 to +0.98 points |
| ruOS, tweets | 57.03% | 58.59% | 59.18% | 6.78 points (61/900) | −4.30 to −0.20 points |

The gate required a two point advantage above every compact control and at most one point of lost sentiment accuracy on each population. Both hosts missed the gate. Neither met the simultaneous confidence bounds. The reference gate lost one of 400 in calibration. Its loss on tweets shows the limit of calibrating exclusively on Reddit. A separately [registered target domain calibration test](target/README.md) recovered 101 of 103 lost sentiment answers on 1,800 fresh tweets. Its stricter gate also reduced anger recognition.

## Official TweetEval emotion benchmark

This separate endpoint predicts anger, joy, optimism and sadness on all 1,421 official test tweets. It starts with a dedicated Cardiff emotion checkpoint and lets the received readout override to anger. Both CPUs used identical source files and ordered examples, with separate feature extraction, fitting and calibration. Hardware dependent numerical differences can change predictions. The two executions reuse the same test examples.

| Macro F1 (0 to 100) | Reference CPU | ruOS CPU |
| --- | ---: | ---: |
| Cardiff emotion specialist | 80.15528 | 80.27780 |
| Specialist plus WSP | **80.31563** | **80.38611** |
| Specialist plus conventional soft distillation | 80.31563 | **80.44032** |
| WSP gain against intact specialist, paired 95% interval | +0.16035 [0.00000, 0.37215] | +0.10831 [0.00000, 0.27289] |

The new output corrected three reference predictions and two ruOS predictions. On the reference CPU, WSP and distillation made the same 1,421 decisions. On ruOS, distillation corrected one additional tweet. The benchmark authors' evaluation script reproduced every arm's macro F1 on each archive. The independent hosts do not constitute independent statistical samples.

The original comparison selected the published TimeLMs 2021 score of 80.2. A stronger published result identified after registration is 81.5 for QA-BPE-nlp + BERTweet. Both use the official four class benchmark. These are external, unpaired point estimates. Neither establishes an advantage under our communication budget. The observed results do not establish a state of the art accuracy claim. [Official table and evaluator](https://github.com/cardiffnlp/tweeteval/tree/4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66) · [QA-BPE Table 21](https://arxiv.org/html/2602.06394v1).

## What crossed the link

The sender uses 47 whitened receiver coordinates and a bias. The candidate sends 48 signed int8 coefficients and one float32 scale. The payload is 52 bytes, the signed body is 154 bytes and one host C Air frame is 218 bytes. The strongest direct control spends the same 52 bytes on 13 float32 scalars. Conventional distillation uses the same 416 teacher responses, 160 public gold examples, protection examples and parameter codec as WSP. The teacher direction and conventional distillation require fitting. The WSP readout has a closed form update.

The 218 byte measurement assumes both sides have the pinned models, setup, ordered public texts and verification key. The reference setup contains 34,448 bytes of anchor IDs and text, 295,288 bytes of receiver basis values, 18,440 bytes of source direction and 38,403 bytes of public local and protection text, IDs and labels. Storing precomputed shared fit features at the sender costs another 297,792 bytes. The benchmark's dedicated emotion specialist is an additional model. Physical radio delivery and general knowledge transfer to arbitrary model families remain untested.

## Apply the portable addition

The independent installer wraps a 6,760 byte ONNX addition in a 9,677 byte JSON artifact. It binds the complete base model, appended branch and reconstructed model by SHA256. It never overwrites a destination. The portable artifact contains the folded coefficients and calibrated receiver threshold, so installation requires only the pinned target checkpoint and ONNX.

```sh
python -m pip install onnx==1.19.1
python harness/wsp/frontier/artifact.py apply /path/to/target-model.onnx \
  harness/wsp/frontier/reference/adapter.json \
  f825228564f424efcb35e28bb52f38972e29d73425b88eb714d7c97b4aca99d3 \
  /path/to/new-receiver.onnx
```

The reference artifact adds six pure nodes and four constants. Its checker compares all 1,551 original nodes, 355 parameter tensors, original inputs, outputs and operator versions. The original `logits` output remains available. `lmwsp_score` and `lmwsp_route` expose the new score and automatic decision separately. Tests on 64 texts per host found zero native logit drift and zero differences between compiled and standalone routing. The reference installation reproduced exactly the graph with SHA256 `f25976b3661e943fc52ccdc9ad2610e1972bb6b7a8ac3321019e8967379fe5bc`. An application that consumes automatic routing can still lose sentiment decisions as measured above. [Pinned base model](https://huggingface.co/Xenova/twitter-roberta-base-sentiment-latest/blob/f3ec4d0925f90c3ca7ee7814f52d6ee7cf180445/onnx/model_quantized.onnx) · [Portable addition](reference/adapter.json).

## Replay and provenance

The protocol preceded candidate tuning. The code, reference policies, mapping and patch hashes were fixed before opening the new test scores. The specialist's config output order disagreed with the mapping used by its model card. A one to one mapping fitted on 96 disjoint validation examples gave 78 correct and was frozen before scoring. The official test had zero exact normalized development text overlaps. The public training history of these pretrained checkpoints cannot rule out upstream test exposure. [Registered protocol](https://github.com/ruvnet/LatentMesh/blob/c067edbc3d852e2c7f4b8f84fda6954845ca7420/harness/wsp/frontier/protocol.md) · [Frozen reference weights and policies](https://github.com/ruvnet/LatentMesh/blob/7d79d0cdee232996fe46e759066af533869accdb/harness/wsp/frontier/lock.json) · [Cardiff model usage](https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion).

From the LatentMesh repository and the pinned sibling ruv-FANN checkout in the [parent instructions](../README.md), use Python 3.12, Node 22 and a C11 compiler. The first two checks use the committed predictions, without downloading models. The third re-extracts all features and runs the signed C transport, WSP patch, scoring and compiled graph checks. Its second argument must name a fresh directory.

```sh
python -m pip install -r harness/wsp/language/requirements.txt
python harness/wsp/frontier/evidence.py verify harness/wsp/frontier
python harness/wsp/frontier/test_artifact.py
bash harness/wsp/frontier/replay.sh /tmp/latentmesh-assets /tmp/latentmesh-run
```

To rerun the benchmark authors' evaluator, use their pinned script, unchanged test labels and each committed prediction archive.

```sh
git clone https://github.com/cardiffnlp/tweeteval.git /tmp/tweeteval
git -C /tmp/tweeteval checkout 4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66
for host in reference ruos; do
  python harness/wsp/frontier/check_official.py "harness/wsp/frontier/$host" \
    /tmp/tweeteval/evaluation_script.py /tmp/tweeteval/datasets/emotion/test_labels.txt
done
```

[Reference scorecard](reference/decision.json) · [ruOS scorecard](ruos/decision.json) · [Reference observations](reference/observations.archive.json) · [ruOS observations](ruos/observations.archive.json).
