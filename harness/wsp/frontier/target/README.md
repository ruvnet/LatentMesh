# Target domain calibration protected sentiment at the cost of anger recall

On 1,800 fresh sentiment tweets, the original hybrid displaced 103 correct answers. The calibrated hybrid displaced two. That recovered 101 answers, or 98.1% of the original losses. The loss fell from 5.72 to **0.11 percentage points**, with a simultaneous one sided 95% upper bound of **0.88**. This passed the registered one point protection requirement on the reference CPU.

The original sentiment model answered 1,272 of the 1,800 cases correctly. In the calibrated arm, both displaced answers belonged to the negative sentiment class. The rule raised the anger threshold from 0.231885 to 0.759935 using 600 independent target domain sentiment labels. All methods received the same additional supervision. The readout weights and 218 byte frame stayed fixed.

| Fresh test | Examples | Original WSP | Target calibrated WSP | Best calibrated compact control |
| --- | ---: | ---: | ---: | ---: |
| GoEmotions anger vs other negative emotions | 150 | 69.33% | 52.67% | Four bit binary, 52.67% |
| CARER anger vs sadness | 512 | 50.00% | 50.00% | All compact controls tied at 50.00% |
| Correct sentiment answers displaced | 1,800 | 103 | 2 | One for soft distillation or the 13 float control, zero for an intact receiver |

On Reddit the hybrid went from correctly detecting 35 of 75 anger cases to four of 75. False positives fell from six to zero. The paired 95% interval for its change in balanced accuracy was −23.33 to −10.67 points. On CARER the hybrid produced zero anger decisions for 256 anger and 256 sadness cases under either rule. Abstaining explained its 50% accuracy. The zero point advantage against the best control on both skill populations missed the two point advancement requirement. The retention result applies to the balanced Twitter sentiment population and this host.

The 900 tweets used in the preceding evaluation were excluded by the original cohort fingerprint. Calibration and evaluation used 200 and 600 new examples per sentiment class. The independent CARER study used the official test split, excluded exact normalized overlap with GoEmotions and all the TweetEval texts used here, and required no CARER examples during fitting. The small Reddit population reflects the eligible single label dev examples after excluding 64 earlier anchors. The authors provide CARER for educational and research use. Public pretrained model exposure is unknown. [CARER paper](https://aclanthology.org/D18-1404/) · [Dataset and revision](https://huggingface.co/datasets/dair-ai/emotion/tree/cab853a1dbdf4c42c2b3ef2173804746df8825fe).

The threshold calibration is a separately registered follow up on the reference host. It does not alter the prior two host results or the official four class scores. Retain the standalone output for experiments. An automatic override would need evidence that it recognizes anger across domains at the required protection level. The score diagnostic below narrows the next comparison to domain calibration and feature quality.

## What the scores still recognize

After the registered test, we inspected the ranking of the frozen scores without selecting a new threshold. On CARER, the hybrid ranked an anger example above a sadness example in 72.17% of pairs (AUROC 0.72173, approximate 95% interval 0.67782 to 0.76563). The median anger and sadness scores were −0.63155 and −0.90280. Both sat below the original override threshold of +0.23189. On Reddit the AUROC was 0.84516 (0.78004 to 0.91027). Conventional distillation and 13 floats scored 0.72780 and 0.73080 on CARER. The hybrid's all negative decisions concealed useful ranking information shared by the simpler methods.

This post hoc diagnostic supports comparing a target domain score mapping against a new sender direction on an untouched corpus. Calibration must demonstrate both protected sentiment and improved anger decisions at the same operating point. No test threshold was selected. [Per method rank statistics](ranking.json).

## Replay

To verify the published result from categorical predictions, use NumPy and SciPy from the parent requirements. To rerun inference, first execute the parent [replay](../README.md) to recreate the fitted receiver state, then use its cache and output directory below. DuckDB reads the pinned 129 kB CARER Parquet file locally. The original text is never needed for the archive replay.

```sh
python harness/wsp/frontier/target/summarize.py verify harness/wsp/frontier/target
python -m pip install duckdb==1.4.4
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python harness/wsp/frontier/target/trial.py calibrate /tmp/latentmesh-assets /tmp/latentmesh-run /tmp/latentmesh-target
python harness/wsp/frontier/target/trial.py evaluate /tmp/latentmesh-assets /tmp/latentmesh-run /tmp/latentmesh-target
python harness/wsp/frontier/target/summarize.py score /tmp/latentmesh-target
python harness/wsp/frontier/target/diagnose.py /tmp/latentmesh-assets /tmp/latentmesh-run /tmp/latentmesh-target
```

[Prospective protocol](https://github.com/ruvnet/LatentMesh/blob/4864525f64fd5145be82bd74a2be9a33780cba64/harness/wsp/frontier/target/protocol.md) · [Calibration lock before scoring](https://github.com/ruvnet/LatentMesh/blob/0dd803cb08b6b0b7f1c9aec058392ffc67bc489b/harness/wsp/frontier/target/lock.json) · [Scorecard](decision.json) · [Paired predictions](observations.archive.json).
