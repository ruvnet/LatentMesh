# Does a dialogue trained sender improve the local anger detector?

Keep the existing local receiver. Adding the full precision EmoBERTa anger score improved F0.5 by **1.41 percentage points** over an equally trained local receiver on 4,096 reserved conversations. The paired 95% interval was **−1.17 to +3.93**. Both CPUs made the same primary decisions. The required lower bound was strictly above zero, so we stopped before fitting or sending a new 218 byte program. A separate 4,096 conversations remain unscored.

## Primary outcome

| Receiver at the registered cutoff | True alerts | False alerts | Precision | Recall | F0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched local plus EmoBERTa | 185 | 20 | 90.24% | 21.74% | 55.36% |
| Matched local | 182 | 27 | 87.08% | 21.39% | 53.94% |
| Existing local receiver | 196 | 30 | 86.73% | 23.03% | 55.84% |

Of 851 angry final messages, the sender recovered 27 that matched local missed and lost 24 it had caught. Of the other messages it added four false alerts and removed eleven. Across the 66 changed decisions that yields three additional true alerts and seven fewer false alerts. The gain versus the existing local receiver was −0.48 points (95% interval −3.16 to +2.10).

At the separately registered, stricter validation cutoff, the sender found 125 angry conversations versus 90 for matched local, with nine false alerts in each arm on the reference CPU. The secondary F0.5 gain was +8.97 points (95% interval +5.57 to +12.48), at 14.69% recall. Average precision across all score thresholds was 71.95% for the augmented receiver, 69.48% for matched local, 70.31% for the incumbent and 60.24% for the sender alone. This suggests a potential use for a small number of high precision alerts. Testing that use requires a separately specified recall requirement, independent data and a comparison with the best local cutoff. [Reference scores](reference/decision.json)

## What the trial covers

The published EmoBERTa classifier reads the current utterance. The local Twitter sentiment receiver also reads the preceding turn. Both use RoBERTa base. We added the unrestricted sender margin as the 66th input to the existing closed form ruv-FANN WSP readout. Matched local uses zero for that input. Both use the same 741 DailyDialog fitting labels and 5,465 calibration labels. The rich sender has additional supervised training on MELD and IEMOCAP. This experiment measures incremental information across two checkpoints in one architecture family. [Author implementation](https://github.com/tae898/erc/tree/faf25370d8805a1975bda768cb94288d5d72c490) · [Pinned solver](https://github.com/ruvnet/ruv-FANN/tree/1abb28d3d23f00c04810f59ff7b3797b57bdc24c/packages/wsp)

We resolved all 23,747 supervised training and validation records to the authors’ transcripts. We rejected 1,984 conversations for exact normalized overlap with sender supervision, screened both model inputs against earlier experiments, and required unique normalized inputs across both panels. Of 30,160 official EmoContext training conversations, 15,964 remained eligible. We used 4,096 as a prospective research holdout and checked all 4,096 gold labels against a second distribution. Anger prevalence was 20.78% in this panel. Generalization to live traffic and other architectures is open. The text audit cannot identify paraphrases or exposure during backbone pretraining. [Cohort audit](audit.json) · [Official task](https://aclanthology.org/S19-2005/)

We registered the screening rule and the conditional packet test before fitting. We committed both sets of coefficients, operating cutoffs and panel hashes before scoring any primary labels. The confidence interval uses 10,000 paired conversation resamples. [Protocol at registration](https://github.com/ruvnet/LatentMesh/blob/08e30a696a7915a18d59fe44d64cf5319ad88c06/harness/wsp/frontier/confidence/dialogue/protocol.md) · [Frozen coefficients](https://github.com/ruvnet/LatentMesh/blob/d1957b5f794b2da7d7e50f2e2de51dd21d4e6624/harness/wsp/frontier/confidence/dialogue/lock.json)

## Cost and independent execution

The additional sender requires 498,686,637 checkpoint bytes, 2,108,545 tokenizer bytes and 1,024 configuration bytes. It runs once per current utterance. With batches of 16 and four CPU threads, the sender added 34.35 seconds for 4,096 cases on the reference host and 92.66 seconds on ruOS, excluding checkpoint load. Those are amortized costs of 8.39 and 22.62 ms per conversation. The existing 125,905,426 byte local encoder was present in both arms. These are offline batch measurements. [Reference timing](reference/timing.json) · [ruOS timing](ruos/timing.json)

The independent ruOS run reproduced all primary predictions and the +1.41 point estimate. One false alert at the secondary cutoff differed between hosts. The bootstrap resamples joint prediction patterns across all registered methods and cutoffs, so this changed the ruOS primary interval slightly to −1.15 to +3.94. Both executions evaluate the same conversations. [ruOS scores](ruos/decision.json)

## Reproduce the decision

From the repository root with Python 3.12 and NumPy 2.3.5, replay both archived sets of predictions. The output should show two false packet decisions, zero primary tag differences and one secondary transfer tag difference.

```sh
r=harness/wsp/frontier/confidence/dialogue
python "$r/test_dialogue.py"
OPENBLAS_NUM_THREADS=1 python "$r/dialogue_score.py" compare "$r/reference" "$r/ruos"
```

To rerun model inference, install the pinned CPU dependencies and choose a new output directory. The wrapper verifies source and target data hashes, reconstructs the prospective split, runs both checkpoints and recomputes the decision.

```sh
python -m pip install -r harness/wsp/language/requirements.txt duckdb==1.4.4
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu
python -m pip install transformers==4.57.3
bash "$r/replay.sh" ./model-cache ./fresh-dialogue
```

To refit the matched weights with the pinned ruv-FANN checkout alongside LatentMesh, regenerate the original development features. Their hash and the projection are checked before fitting.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python "$r/../context/develop.py" prepare ./model-cache ./context-fit
mkdir ./dialogue-fit
python "$r/dialogue_data.py" ./model-cache ./dialogue-fit/audit.json
python "$r/train_dialogue.py" prepare ./model-cache ./context-fit ./dialogue-fit
node "$r/../dailydialog/fit.mjs" ./dialogue-fit --matched
python "$r/train_dialogue.py" freeze ./context-fit ./dialogue-fit
cmp "$r/lock.json" ./dialogue-fit/lock.json
```
