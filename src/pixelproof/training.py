"""Fixed, reproducible scikit-learn experiment and safe JSON bundle writer."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

_SEED = 42
_TEST_FRACTION = 0.25
_CLASSES = list(range(10))
_MODEL_CONFIG = {
    "dataset": "sklearn.load_digits",
    "estimator": "LogisticRegression",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "tol": 1e-4,
    "seed": _SEED,
    "test_fraction": _TEST_FRACTION,
    "preprocessing": "divide_by_16",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_sha256(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _probability_rows(values: np.ndarray) -> list[list[float]]:
    rows = [[float(probability) for probability in row] for row in values]
    if any(not math.isfinite(probability) for row in rows for probability in row):
        raise RuntimeError("training produced a non-finite probability")
    return rows


def build_bundle() -> dict:
    """Train the fixed classifier and return its model, holdout, and evaluation."""
    dataset = load_digits()
    pixels_array = dataset.data.astype(np.int64)
    labels_array = dataset.target.astype(np.int64)
    if pixels_array.shape != (1797, 64) or labels_array.shape != (1797,):
        raise RuntimeError("load_digits returned an unexpected dataset shape")
    if int(pixels_array.min()) != 0 or int(pixels_array.max()) != 16:
        raise RuntimeError("load_digits pixel range differs from the fixed contract")

    all_pixels = pixels_array.tolist()
    all_labels = labels_array.tolist()
    dataset_sha256 = _json_sha256({"pixels": all_pixels, "labels": all_labels})

    train_raw, test_raw = train_test_split(
        np.arange(len(labels_array)),
        test_size=_TEST_FRACTION,
        random_state=_SEED,
        stratify=labels_array,
    )
    train_indices = sorted(int(index) for index in train_raw)
    test_indices = sorted(int(index) for index in test_raw)
    if set(train_indices) & set(test_indices) or sorted(train_indices + test_indices) != list(
        range(len(labels_array))
    ):
        raise RuntimeError("train/test split is not a complete disjoint partition")
    train_indices_sha256 = _json_sha256(train_indices)
    test_indices_sha256 = _json_sha256(test_indices)

    estimator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        tol=1e-4,
        random_state=_SEED,
    )
    train_x = pixels_array[train_indices].astype(np.float64) / 16.0
    test_x = pixels_array[test_indices].astype(np.float64) / 16.0
    train_y = labels_array[train_indices]
    test_y = labels_array[test_indices]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(train_x, train_y)
    did_not_converge = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    if did_not_converge or np.any(estimator.n_iter_ >= 1000):
        raise RuntimeError("LogisticRegression did not converge within max_iter=1000")
    if estimator.classes_.tolist() != _CLASSES:
        raise RuntimeError("trained estimator did not learn all ten digits in canonical order")

    coef = [[float(value) for value in row] for row in estimator.coef_]
    intercept = [float(value) for value in estimator.intercept_]
    if any(not math.isfinite(value) or abs(value) > 100 for row in coef for value in row):
        raise RuntimeError("trained coefficients exceed the artifact bounds")
    if any(not math.isfinite(value) or abs(value) > 100 for value in intercept):
        raise RuntimeError("trained intercepts exceed the artifact bounds")

    training_metadata = {
        "dataset": "sklearn.load_digits",
        "dataset_sha256": dataset_sha256,
        "train_indices_sha256": train_indices_sha256,
        "seed": _SEED,
        "test_fraction": _TEST_FRACTION,
        "estimator": "LogisticRegression",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "tol": 1e-4,
        "sklearn_version": str(sklearn.__version__),
        "numpy_version": str(np.__version__),
    }
    model = {
        "schema_version": 1,
        "feature_count": 64,
        "classes": _CLASSES.copy(),
        "pixel_max": 16,
        "preprocessing": "divide_by_16",
        "coef": coef,
        "intercept": intercept,
        "training": training_metadata,
    }
    model_bytes = _canonical_bytes(model)
    model_sha256 = _sha256(model_bytes)

    predictions = estimator.predict(test_x).astype(int).tolist()
    probability_values = estimator.predict_proba(test_x)
    probabilities = _probability_rows(probability_values)
    test_cases = [
        {
            "id": int(index),
            "pixels": all_pixels[index],
            "label": all_labels[index],
        }
        for index in test_indices
    ]
    test_predictions = [
        {
            "id": int(index),
            "label": int(label),
            "predicted": int(predicted),
            "probabilities": probability,
        }
        for index, label, predicted, probability in zip(
            test_indices, test_y.tolist(), predictions, probabilities, strict=True
        )
    ]
    accuracy = float(accuracy_score(test_y, predictions))
    macro_f1 = float(
        f1_score(test_y, predictions, labels=_CLASSES, average="macro", zero_division=0)
    )
    heldout_log_loss = float(log_loss(test_y, probability_values, labels=_CLASSES))

    precision, recall, digit_f1, support = precision_recall_fscore_support(
        test_y, predictions, labels=_CLASSES, zero_division=0
    )
    per_digit = [
        {
            "digit": digit,
            "precision": float(precision[digit]),
            "recall": float(recall[digit]),
            "f1": float(digit_f1[digit]),
            "support": int(support[digit]),
        }
        for digit in _CLASSES
    ]
    train_counts = np.bincount(train_y, minlength=10)
    majority_digit = min(_CLASSES, key=lambda digit: (-int(train_counts[digit]), digit))
    baseline_predictions = np.full_like(test_y, majority_digit)
    baseline = {
        "selection": "training majority class, ties lowest digit",
        "majority_digit": int(majority_digit),
        "accuracy": float(accuracy_score(test_y, baseline_predictions)),
        "macro_f1": float(
            f1_score(
                test_y, baseline_predictions, labels=_CLASSES, average="macro", zero_division=0
            )
        ),
    }

    zero_pixels = [0] * 64
    full_pixels = [16] * 64
    checkerboard = [16 * ((row + col) % 2) for row in range(8) for col in range(8)]
    probe_pixels = [
        ("all_zero", zero_pixels),
        ("all_16", full_pixels),
        ("checkerboard", checkerboard),
    ]
    probe_probabilities = _probability_rows(
        estimator.predict_proba(
            np.asarray([pixels for _, pixels in probe_pixels], dtype=np.float64) / 16.0
        )
    )
    stress_probes = [
        {
            "name": name,
            "pixels": pixels,
            "predicted_digit": int(np.argmax(probability)),
            "probabilities": probability,
        }
        for (name, pixels), probability in zip(probe_pixels, probe_probabilities, strict=True)
    ]

    evaluation = {
        "model_sha256": model_sha256,
        "dataset_sha256": dataset_sha256,
        "train_indices_sha256": train_indices_sha256,
        "test_indices_sha256": test_indices_sha256,
        "test_cases_sha256": _json_sha256(test_cases),
        "config": _MODEL_CONFIG.copy(),
        "converged": True,
        "train_count": len(train_indices),
        "test_count": len(test_indices),
        "train_class_counts": [int(count) for count in train_counts],
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "log_loss": heldout_log_loss,
        },
        "baseline": baseline,
        "per_digit": per_digit,
        "confusion_matrix": confusion_matrix(test_y, predictions, labels=_CLASSES)
        .astype(int)
        .tolist(),
        "predictions": test_predictions,
        "misclassified_ids": [
            item["id"] for item in test_predictions if item["label"] != item["predicted"]
        ],
        "error_count": sum(item["label"] != item["predicted"] for item in test_predictions),
        "stress_probes": stress_probes,
    }
    return {
        "model": model,
        "evaluation": evaluation,
        "split": {"train": train_indices, "test": test_indices},
        "test_cases": test_cases,
    }


def write_bundle(output: Path) -> dict:
    """Write a new candidate bundle directory without overwriting prior evidence."""
    if not isinstance(output, Path):
        raise TypeError("output must be a pathlib.Path")
    target = output
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"output directory already exists: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"output parent directory does not exist: {target.parent}")

    bundle = build_bundle()
    model_bytes = _canonical_bytes(bundle["model"])
    digest = _sha256(model_bytes)
    if digest != bundle["evaluation"]["model_sha256"]:
        raise RuntimeError("bundle model digest differs from its evaluation metadata")
    files = {
        "model.json": model_bytes,
        "model.sha256": (digest + "\n").encode("ascii"),
        "evaluation.json": _pretty_bytes(bundle["evaluation"]),
        "split.json": _pretty_bytes(bundle["split"]),
        "test_cases.json": _pretty_bytes(bundle["test_cases"]),
    }
    target.mkdir(exist_ok=False)
    created: list[Path] = []
    try:
        for name, content in files.items():
            file_path = target / name
            with file_path.open("xb") as handle:
                created.append(file_path)
                handle.write(content)
    except BaseException:
        for file_path in created:
            file_path.unlink(missing_ok=True)
        target.rmdir()
        raise
    return bundle
