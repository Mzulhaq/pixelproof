# PixelProof interview guide

PixelProof packages a trained multinomial logistic regression model as a validated JSON artifact and serves predictions through a bounded Python API. Scikit-learn and NumPy are used for training only; serving inference uses pure Python. The project is about explicit model packaging, data validation, and evaluation, not about creating a novel classifier or building a generative AI system.

## Architecture questions and notes

### What is stored in the model artifact?

The JSON artifact has a strict schema: schema version, feature count, ordered classes, pixel maximum, preprocessing identifier, coefficient and intercept arrays, and training metadata. Dimensions and numeric bounds are checked, duplicate JSON keys and non-finite numbers are rejected, and boolean values cannot stand in for numbers. Artifact size is limited. The model stores fixed parameters rather than executable Python objects, so loading does not require pickle or joblib.

### What does the checksum establish?

The model ID is the SHA-256 digest of the exact artifact bytes. A supplied expected digest is checked before the artifact is accepted, and the bundled sidecar records the digest. This detects an accidental or unexpected byte mismatch when the expected value is trusted. A checksum is not a digital signature and does not prove who created the file or that its source is trustworthy.

### How does serving produce a prediction?

Each input is exactly 64 integer pixel values from 0 through 16, excluding booleans. Values are normalized by 16. The service computes each class logit from the stored coefficient row and intercept, then uses a stable softmax and returns the highest-scoring digit with all class probabilities. Ties select the lowest digit. Batch input accepts 1–32 images and validates the entire batch before computing predictions. Runtime inference does not import scikit-learn or NumPy.

### What limits apply to the HTTP interface?

The read-only API exposes health, model metadata, and prediction routes. Prediction requests must be JSON and are streamed under a 32 KiB body limit, even without a useful Content-Length header. Invalid media types, oversize bodies, invalid JSON, and schema or pixel errors have separate responses. The default bind is loopback. The service has no authentication, rate limits, multi-tenant access controls, external calls, arbitrary artifact upload, or model reload endpoint.

### How was the model trained and compared?

The experiment uses the 1,797-sample scikit-learn `load_digits` bundle: flattened 8×8 images with integer pixels 0–16 and labels 0–9. The fixed split is stratified, with 25% held out and random seed 42; index lists are sorted for reproducibility. Multinomial `LogisticRegression` uses `C=1.0`, `solver='lbfgs'`, `max_iter=1000`, and `tol=1e-4`, trained on pixels divided by 16. A majority-class baseline selects its class from training counts only, breaking ties toward the lower digit. The held-out set is not used for preprocessing fit, tuning, threshold choice, or retraining.

The model artifact's pure-Python probabilities are checked against scikit-learn on every held-out sample with absolute tolerance `1e-10`. Evaluation reports held-out accuracy, macro-F1, multiclass log loss, per-digit precision/recall/F1/support, confusion matrix, per-case predictions and probabilities, and every misclassified sample ID. The majority baseline reports accuracy and macro-F1; it does not receive an invented log-loss score. Fixed all-zero, all-16, and checkerboard stress inputs are reported separately without assigning them correctness labels. See [the full evaluation](evaluation.json) for all 450 predictions and 17 mistakes.

### What can the returned probabilities tell us?

They are uncalibrated model outputs, not guarantees that a prediction is correct. The API always chooses a digit for a valid pixel array; it has no abstention or out-of-distribution detector. Inputs such as blank or saturated images may satisfy the numeric schema while having no meaningful relationship to handwritten digits. The service cannot determine that they are outside the training distribution.

### What does the dataset split mean?

Scikit-learn documents `load_digits` as a copy of the UCI dataset's test set. This project resplits that bundled copy for a fixed experiment, so the resulting scores are not the official UCI test benchmark. The bundle has no writer IDs, so a random stratified split does not support a claim of writer-independent generalization. See [dataset attribution](../data/README.md).

## 30-minute exercise: fixed occlusion probe

Add a separate evaluation probe that masks a fixed 2×2 pixel square at a chosen location on each held-out image. Keep the original split and `test_cases.json` unchanged. For the new probe, derive the modified arrays from the frozen held-out cases, predict with the frozen model, and write a separate result file with its own probe description and input hash. Compare accuracy and per-digit errors with the unmodified held-out cases without changing the training set, model, thresholds, or original evaluation results.

Hints: use a copy of each 8×8 row before zeroing the selected square; do not mutate shared test data. Record coordinates and fill value in the probe metadata. An answer review should check repeatability, confirm the original model ID and test IDs are retained, and describe the observed change without claiming robustness from one small perturbation.

## Resume drafts

These drafts describe repository features, not personal impact. Adapt them only after confirming they accurately reflect your contribution and you can explain the implementation.

- Packaged a trained multinomial logistic regression classifier as a schema-validated, checksummed JSON artifact and served bounded predictions through a pure-Python inference API.
- Compared a fixed stratified holdout model with a training-only majority baseline using classification metrics, per-digit error analysis, and frozen-artifact parity checks.

AI coding assistance contributed to this project. Be transparent about that assistance and describe only work you personally reviewed, understood, and can demonstrate. Do not claim to have authored the model training or achieved particular metrics unless the artifacts and your actual contribution support that claim.
