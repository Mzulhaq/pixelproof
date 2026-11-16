from __future__ import annotations

import hashlib
import json
import math

import pytest

from pixelproof.model import Model
from scripts.evaluate import verify_evaluation


def canonical_sha(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def fixture_bundle():
    train = list(range(1347))
    test = list(range(1347, 1797))
    training = {
        "dataset": "sklearn.load_digits",
        "dataset_sha256": "a" * 64,
        "train_indices_sha256": canonical_sha(train),
        "seed": 42,
        "test_fraction": 0.25,
        "estimator": "LogisticRegression",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "tol": 1e-4,
        "sklearn_version": "1.8.0",
        "numpy_version": "2.0.0",
    }
    artifact = {
        "schema_version": 1,
        "feature_count": 64,
        "classes": list(range(10)),
        "pixel_max": 16,
        "preprocessing": "divide_by_16",
        "coef": [[0.0] * 64 for _ in range(10)],
        "intercept": [0.0] * 10,
        "training": training,
    }
    model_bytes = json.dumps(
        artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    model = Model.from_bytes(model_bytes)
    pixels = [0] * 64
    test_cases = [{"id": index, "pixels": pixels.copy(), "label": 7} for index in test]
    split = {"train": train, "test": test}
    probabilities = [0.1] * 10
    predictions = [
        {"id": index, "label": 7, "predicted": 0, "probabilities": probabilities.copy()}
        for index in test
    ]
    confusion = [[0] * 10 for _ in range(10)]
    confusion[7][0] = len(test)
    per_digit = [
        {
            "digit": digit,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": len(test) if digit == 7 else 0,
        }
        for digit in range(10)
    ]
    probes = []
    probe_pixels = {
        "all_zero": [0] * 64,
        "all_16": [16] * 64,
        "checkerboard": [16 * ((row + column) % 2) for row in range(8) for column in range(8)],
    }
    for name, probe in probe_pixels.items():
        result = model.predict(probe)
        probes.append(
            {
                "name": name,
                "pixels": probe,
                "predicted_digit": result["digit"],
                "probabilities": result["probabilities"],
            }
        )
    evaluation = {
        "model_sha256": model.model_id,
        "dataset_sha256": training["dataset_sha256"],
        "train_indices_sha256": training["train_indices_sha256"],
        "test_indices_sha256": canonical_sha(test),
        "test_cases_sha256": canonical_sha(test_cases),
        "config": {
            "dataset": "sklearn.load_digits",
            "estimator": "LogisticRegression",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "tol": 1e-4,
            "seed": 42,
            "test_fraction": 0.25,
            "preprocessing": "divide_by_16",
        },
        "converged": True,
        "train_count": len(train),
        "test_count": len(test),
        "train_class_counts": [len(train), 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "metrics": {"accuracy": 0.0, "macro_f1": 0.0, "log_loss": -math.log(0.1)},
        "baseline": {
            "selection": "training majority class, ties lowest digit",
            "majority_digit": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
        },
        "per_digit": per_digit,
        "confusion_matrix": confusion,
        "predictions": predictions,
        "misclassified_ids": test.copy(),
        "error_count": len(test),
        "stress_probes": probes,
    }
    return model, evaluation, split, test_cases


def test_verifier_accepts_consistent_frozen_evaluation():
    bundle = fixture_bundle()
    summary = verify_evaluation(*bundle)
    assert summary["train_count"] == 1347
    assert summary["test_count"] == 450
    assert summary["accuracy"] == 0.0
    assert summary["error_count"] == 450
    assert summary["max_probability_delta"] == 0.0


@pytest.mark.parametrize(
    "mutation",
    [
        "label",
        "pixels",
        "split",
        "metric",
        "model_hash",
        "boolean_config",
        "boolean_confusion",
        "boolean_probe",
    ],
)
def test_verifier_rejects_tampered_frozen_inputs(mutation):
    model, evaluation, split, test_cases = fixture_bundle()
    if mutation == "label":
        test_cases[0]["label"] = 8
    elif mutation == "pixels":
        test_cases[0]["pixels"][0] = 1
    elif mutation == "split":
        split["test"][0], split["test"][1] = split["test"][1], split["test"][0]
    elif mutation == "metric":
        evaluation["metrics"]["accuracy"] = 0.5
    elif mutation == "model_hash":
        evaluation["model_sha256"] = "0" * 64
    elif mutation == "boolean_config":
        evaluation["config"]["C"] = True
    elif mutation == "boolean_confusion":
        evaluation["confusion_matrix"][0][0] = False
    else:
        evaluation["stress_probes"][0]["pixels"][0] = False
    with pytest.raises(ValueError):
        verify_evaluation(model, evaluation, split, test_cases)


def test_verifier_recomputes_metrics_instead_of_trusting_report():
    model, evaluation, split, test_cases = fixture_bundle()
    evaluation["confusion_matrix"][7][0] -= 1
    evaluation["confusion_matrix"][7][1] += 1
    with pytest.raises(ValueError, match="confusion"):
        verify_evaluation(model, evaluation, split, test_cases)
