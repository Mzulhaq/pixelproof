# Validation evidence

## Local results — November 16, 2026

Executed on Windows with Python 3.11.9 using the pinned dependency sets. These are actual checks, not estimated outcomes.

| Command / check | Observed result |
| --- | --- |
| `python -m pip install -r requirements.lock` in a new virtual environment | Installed successfully |
| `python -m pip install --no-deps dist/pixelproof-0.1.0-py3-none-any.whl` | Clean wheel install succeeded; packaged model loaded |
| `python -m pip check` | No broken requirements in training and clean serving environments |
| `python -m pytest --cov=pixelproof --cov-report=term-missing` with training extras | 77 passed; 92% statement coverage (318 package statements, 27 missed) |
| `python -m pytest -q` in the clean serving environment | 65 passed; training test module skipped because NumPy was absent |
| `python scripts/demo.py` in both environments | Actual loopback HTTP health/metadata/batch prediction and 422 rejection checks passed; server stopped |
| `python scripts/evaluate.py` in both environments | All 450 frozen cases verified; 17 errors; maximum probability difference from sklearn report `5.551115123125783e-16` |
| `python scripts/train.py candidate` | Wrote a new five-file bundle; 1,347 train / 450 test; convergence confirmed |
| `python -m build` | Built source distribution and platform-independent wheel; wheel includes model JSON, digest and attribution |
| `python -m ruff check .` and `python -m ruff format --check .` | Passed |
| Documented CLI `predict`, `serve` and real HTTP request | Passed in the clean wheel environment |
| Publication scan and Markdown links | 40 staged files checked; no credential/private path patterns found; local documentation links resolve |

The clean serving environment explicitly confirmed that neither `numpy` nor `sklearn` was installed. Training tests check the committed split, holdout pixels/labels, source dataset hash and baseline training counts against a new run using scikit-learn's bundled source data. Pure-Python predictions agree with the trained sklearn estimator for every held-out example within absolute `1e-10` (relative tolerance disabled).

Recomputed metrics: accuracy `0.9622222222222222`, macro-F1 `0.9619897319063684`, log loss `0.1615452096875058`; training-majority baseline accuracy `0.10222222222222223`, macro-F1 `0.018548387096774192`. The tiny final log-loss rounding difference from the sklearn report is below the declared tolerance. These scores are for the specified resplit of a small dataset; they are not general handwriting performance or speed measurements.

Independent code and documentation review checked validation, artifact safety, preprocessing, leakage, baseline selection, export parity, all held-out source samples, metric claims and stress probes. The probability-comparison test was tightened to enforce an absolute tolerance; stale draft references were corrected. No unresolved material findings remained.

## Remote checks

Docker is not installed on the local machine. [GitHub Actions run 43884216941](https://github.com/Mzulhaq/pixelproof/actions/runs/43884216941) passed all five jobs for source commit `efc0e81edd3e3a640414fe15c913d175f614cfcb`:

- Python 3.11 on Ubuntu: locked install, dependency check, Ruff, runtime tests, real HTTP demo, frozen evaluation and package build.
- Python 3.13 on Ubuntu: the same serving checks.
- Python 3.11 on Windows: the same serving checks.
- Training on Ubuntu/Python 3.11: 12 training tests passed, followed by a fresh candidate bundle.
- Container: image built successfully; nonroot HTTP demo and frozen evaluation both passed under `docker run --rm --network none`.

The Linux retraining run produced SHA-256 `786174211f03ac4b576aa62858b40df894946cf49d748de1536806cf50760041`, accuracy `0.9622222222222222`, macro-F1 `0.9619897319063684` and log loss `0.16154520968754013`. Its digest differs from the committed Windows-trained artifact because of floating-point coefficient differences. This is why the project separates reproducible data/configuration/metrics from byte-identical retraining. The container's frozen-artifact evaluator reproduced the committed model's exact reported accuracy and error count, with a maximum probability delta of `5.551115123125783e-16`.

The final documentation commit is also verified through the same workflow; use the repository's [Actions history](https://github.com/Mzulhaq/pixelproof/actions) to inspect subsequent commits. No cloud deployment or performance load test is claimed.
