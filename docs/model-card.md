# PixelProof model card

## Purpose and provenance

This fixed multinomial logistic regression demonstrates reproducible model serving on 8×8 digits. It is an educational classifier and does not recognize arbitrary photos, document images or handwriting reliably.

The artifact SHA-256 is `8620b1cbc663a3506e2fecf018ae7eeaf7385297809b04d7bab1f53eb2b41a8b`. Its JSON records dataset and training-index hashes, preprocessing, hyperparameters and training library versions. `data/split.json` preserves the exact partition. The 1,797 samples are from scikit-learn's `load_digits`, itself a copy of the original UCI test set; see [attribution](../data/README.md). This experiment re-splits that bundled data and does not reproduce the official UCI benchmark.

The model was fit on 1,347 samples using fixed `/16` normalization, with 450 held out. Hyperparameters and split were specified before scoring; no parameter search, calibration or threshold selection used the held-out labels. The model converged with the fixed 1,000-iteration cap. The majority baseline uses only training counts and selects digit 3 (137 of 1,347 training labels). The test labels were used only to score the fixed model and baseline and inspect errors.

## Results and mistakes

Held-out accuracy: 433/450 = 96.22%; macro-F1: 0.961990; multiclass log loss: 0.161545. Baseline: 46/450 = 10.22%, macro-F1 0.018548. These describe one small stratified random split with seed 42 and are not a cross-validation estimate or a confidence interval.

| True digit | Support | Correct | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 45 | 44 | 0.9778 | 0.9888 |
| 1 | 46 | 42 | 0.9130 | 0.9032 |
| 2 | 44 | 44 | 1.0000 | 1.0000 |
| 3 | 46 | 46 | 1.0000 | 0.9892 |
| 4 | 45 | 45 | 1.0000 | 0.9677 |
| 5 | 46 | 44 | 0.9565 | 0.9670 |
| 6 | 45 | 43 | 0.9556 | 0.9663 |
| 7 | 45 | 45 | 1.0000 | 0.9890 |
| 8 | 43 | 38 | 0.8837 | 0.9048 |
| 9 | 45 | 42 | 0.9333 | 0.9438 |

Four of the five errors on digit 8 were predicted as 1; the fifth was predicted as 7. Both mistaken 5s were predicted as 9. Two of three errors on 9 were predicted as 4. These are observed confusion counts, not established causal explanations about handwriting shape. Digit 1 has the lowest F1 (0.9032); digit 8 has the lowest recall (0.8837). Perfect recall for several digits on fewer than 50 examples each does not establish universal recognition.

All 17 misclassified sample IDs: `5, 69, 95, 129, 363, 457, 492, 639, 677, 751, 1256, 1264, 1500, 1553, 1573, 1666, 1790`. The full [evaluation](evaluation.json) includes each prediction and probability vector, every per-class precision/recall/F1, and confusion rows (true digit) versus columns (predicted digit).

## Stress probes and uncertainty

These synthetic probes have no true digit and are kept separate from accuracy calculations:

| Input | Predicted digit | Largest probability |
| --- | ---: | ---: |
| All zero | 4 | 0.780776 |
| All 16 | 2 | 0.270613 |
| Alternating 0/16 checkerboard | 7 | 0.406326 |

The blank result illustrates a failure: the model must choose a class even when no digit exists. Softmax probabilities are uncalibrated model scores, not guarantees. The API has no rejection threshold or out-of-distribution detector. Low or high scores should not be converted into a claim that an input is a real digit.

## Reproduction and boundaries

`python scripts/evaluate.py` uses the committed model, verifies holdout/report hashes, recomputes metrics, checks all 450 probability vectors against the recorded sklearn run within `1e-10`, and checks stress probes. It needs no scikit-learn installation. Training tests independently compare exported predictions to a fresh sklearn run and check sample provenance against the original bundled dataset. SHA-256 checks prevent accidental mismatches; they are not cryptographic signatures from a trusted producer.

Run `python scripts/train.py candidate` in the pinned training environment to generate new evidence. Retraining may vary slightly across platforms. The split is random by sample; absent writer IDs prevent claims of writer independence. The dataset is small, historical and low resolution. No demographic analysis, handwriting-product evaluation, production monitoring, performance benchmark or external deployment was performed.
