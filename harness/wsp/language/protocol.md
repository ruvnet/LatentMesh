# Frozen experiment 2 · anger transfer between language encoders

Record before fitting, opening held out labels for scoring, or choosing an operating point. This is a new task on intact public checkpoints. Intact means the receiver keeps all of its original sentiment parameters. The proposed patch controls an additional response that can override them.

## Fixed inputs and partitions

Use the quantized ONNX exports of Xenova/distilbert-base-uncased-finetuned-sst-2-english at 0b6928efcb76139cae2c6881d49cda67fe119f42 and Xenova/twitter-roberta-base-sentiment-latest at f3ec4d0925f90c3ca7ee7814f52d6ee7cf180445. Use the respective native tokenizers, CPU, one example at a time and a maximum of 96 tokens. Read the last classification token from /Gather_output_0 and /classifier/Gather_output_0, both before the native classifier. Check instrumented native decisions against the unmodified graphs.

Use google-research/google-research GoEmotions simplified train, dev and test TSVs and emotions.txt at 2adf640a14f11025ae5a9d0ec493b78530d276d3. Keep singly labeled examples. For duplicate IDs or NFKC normalized, case folded, whitespace collapsed text, retain the earliest occurrence in train, dev, test and source line order. Exclude other partitions' duplicates even when the earlier example was not sampled. No pretrained corpus overlap audit is possible.

The new task distinguishes anger from the pooled annoyance, disappointment, disgust and fear labels. Rank IDs per class by SHA256 of the UTF8 string 'latentmesh-language-v1/' plus the original ID. Take equal counts of positive and pooled negative rows, capped at 256 of each for fitting the sender on train, 32 of each for shared anchors on dev, and 160 of each for final scoring on test. Interleave the two ranked strata, positive first. The separate retention test uses the same ranking on test, equal counts of joy and sadness up to 150 each. Their expected original sentiment outputs are positive and negative respectively. These two populations have disjoint IDs and text. Trainer and receiver compiler never receive test data. The receiver compiler receives ordered anchor text and sender messages, without anchor labels. Anchor selection is balanced by annotation and is a provisioning assumption.

## Fixed signal and receiver

Fit StandardScaler then sklearn LogisticRegression(C=0.01, solver='lbfgs', max_iter=1000) on sender train CLS vectors for anger membership. The signal is tanh(decision_function / 2). Freeze the direction parameters, preprocessing, probe order and receiver feature basis and record their hashes before extracting or scoring test activations. Compare the direction with the sender's native negative class probability on the new task, including AUROC. Native soft signal is twice the negative probability minus one.

Fit 12 component centered PCA using full SVD of all 64 receiver anchor CLS vectors only. The added linear readout starts at zero. Fit its 12 weights and bias using the existing scalar ruv-FANN WSP closed form solver, with identical settings for all learned arms. At inference a strictly positive new score emits anger. Otherwise use the intact receiver's native three way sentiment decision. No fitting of the threshold. For target scoring use balanced binary anger accuracy. For retention score ground truth original sentiment, counting an anger override as an error. Report intact accuracy and paired changes.

## Fixed controls and cost

Use a single local signed packet and 52 available application payload bytes per arm before common headers and signature. Zero pad shorter payloads to the same size. Validate exact bytes and frame count with the real host C Air codec. Separately report useful payload, packet and wire bytes. Provision at most 64 anchors for every arm. All prefix lengths below refer to the same order.

| Arm | Information delivered |
| --- | --- |
| Hybrid | First 48 direction scores quantized to signed bytes with a 4 byte peak scale |
| Hard48 and Hard64 | First 48 or all 64 signs of the identical direction packed as bits, decoded to +1 or -1 |
| Decimal52 | First 52 direction scores rounded to the nearest of ten equally spaced levels on [-1,1], one ASCII digit per score |
| Text4 | All 64 direction scores rounded to 16 equally spaced levels on [-1,1], packed two per byte and base64 encoded |
| Phrase | Literal ASCII 'angry furious enraged\|annoyed afraid disappointed', with a single vertical bar delimiter. The receiver embeds each side using its own checkpoint and fits all 64 anchor cosine similarity differences (positive minus negative) |
| Native | First 48 native negative sentiment scores via the hybrid quantizer |
| Shuffled | The first 48 quantized direction scores permuted with numpy default_rng(72191) before fitting, preserving their multiset |
| Intact | No added response |

Cost reporting includes software loopback framing. Report the exact common anchor text bytes, compact ordered anchor provisioning representation, receiver basis bytes, direction bytes and key and manifest sizes separately from warm transport. Report host timing separately from content hashes. Physical RF performance is unmeasured in this research experiment.

## Decision

Advance to hardware evaluation only if the hybrid's observed balanced accuracy exceeds or equals the strongest of Hard48, Hard64, Decimal52, Text4, Phrase and Native by 2 percentage points AND retained accuracy is within 1 percentage point of the intact receiver. Report paired uncertainty against the observed best comparator with fixed seed 72191, 10000 stratified bootstrap replicates. This interval is descriptive, conditional on the fitted checkpoints and observed comparator. One task and checkpoint pair cannot establish a general transfer advantage. Record a failed gate without tuning or promotion.
