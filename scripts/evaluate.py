"""Verify the frozen evaluation against the packaged dependency-free model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from pixelproof.model import Model

_ROOT = Path(__file__).resolve().parents[1]
_CLASS_COUNT = 10
_PIXEL_COUNT = 64
_DATASET_SIZE = 1797
_TOLERANCE = 1e-10
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_CONFIG = {
    "dataset": "sklearn.load_digits",
    "estimator": "LogisticRegression",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "tol": 1e-4,
    "seed": 42,
    "test_fraction": 0.25,
    "preprocessing": "divide_by_16",
}


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _parse_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON number")
    return number


def _read_json(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique,
            parse_constant=_parse_constant,
            parse_float=_parse_float,
        )
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON in {path}") from error


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_int(
    value: Any, message: str, *, minimum: int | None = None, maximum: int | None = None
) -> int:
    _require(type(value) is int, message)
    if minimum is not None:
        _require(value >= minimum, message)
    if maximum is not None:
        _require(value <= maximum, message)
    return value


def _finite_number(value: Any, message: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value), message)
    return float(value)


def _matches_float(actual: Any, expected: float, field: str) -> None:
    actual_number = _finite_number(actual, f"{field} must be a finite number")
    _require(
        abs(actual_number - expected) <= _TOLERANCE, f"{field} does not match recomputed value"
    )


def _metrics(
    labels: list[int], predictions: list[int], probabilities: list[list[float]]
) -> dict[str, Any]:
    confusion = [[0] * _CLASS_COUNT for _ in range(_CLASS_COUNT)]
    for label, prediction in zip(labels, predictions, strict=True):
        confusion[label][prediction] += 1

    per_digit = []
    f1_values = []
    for digit in range(_CLASS_COUNT):
        true_positive = confusion[digit][digit]
        support = sum(confusion[digit])
        predicted_count = sum(confusion[row][digit] for row in range(_CLASS_COUNT))
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_digit.append(
            {
                "digit": digit,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    accuracy = sum(
        label == prediction for label, prediction in zip(labels, predictions, strict=True)
    )
    accuracy = accuracy / len(labels)
    log_loss = -math.fsum(
        math.log(row[label]) for label, row in zip(labels, probabilities, strict=True)
    ) / len(labels)
    return {
        "accuracy": accuracy,
        "macro_f1": math.fsum(f1_values) / _CLASS_COUNT,
        "log_loss": log_loss,
        "confusion_matrix": confusion,
        "per_digit": per_digit,
        "error_count": sum(
            label != prediction for label, prediction in zip(labels, predictions, strict=True)
        ),
    }


def _validate_split(split: Any) -> tuple[list[int], list[int]]:
    _require(
        isinstance(split, dict) and set(split) == {"train", "test"},
        "split must contain exactly train and test indexes",
    )
    train, test = split["train"], split["test"]
    _require(isinstance(train, list) and isinstance(test, list), "split indexes must be lists")
    for name, indexes in (("train", train), ("test", test)):
        for index in indexes:
            _exact_int(
                index,
                f"{name} indexes must be integer IDs in dataset range",
                minimum=0,
                maximum=_DATASET_SIZE - 1,
            )
        _require(indexes == sorted(indexes), f"{name} indexes must be sorted")
        _require(len(indexes) == len(set(indexes)), f"{name} indexes must be unique")
    _require(not set(train).intersection(test), "train and test indexes overlap")
    _require(
        sorted(train + test) == list(range(_DATASET_SIZE)),
        "train and test indexes must partition all 1797 dataset rows",
    )
    expected_test_count = math.ceil(_DATASET_SIZE * _CONFIG["test_fraction"])
    _require(len(test) == expected_test_count, "test split size must match the fixed 25% holdout")
    return train, test


def _check_probe(model: Model, report_probe: Any, name: str, pixels: list[int]) -> float:
    _require(
        isinstance(report_probe, dict)
        and set(report_probe) == {"name", "pixels", "predicted_digit", "probabilities"},
        f"stress probe {name} has an invalid schema",
    )
    reported_pixels = report_probe["pixels"]
    _require(
        isinstance(reported_pixels, list)
        and all(type(pixel) is int for pixel in reported_pixels)
        and report_probe["name"] == name
        and reported_pixels == pixels,
        f"stress probe {name} input does not match the fixed probe",
    )
    actual = model.predict(pixels)
    predicted = _exact_int(
        report_probe["predicted_digit"],
        f"stress probe {name} prediction must be an integer",
        minimum=0,
        maximum=9,
    )
    _require(predicted == actual["digit"], f"stress probe {name} predicted digit mismatch")
    probabilities = report_probe["probabilities"]
    _require(
        isinstance(probabilities, list) and len(probabilities) == _CLASS_COUNT,
        f"stress probe {name} probabilities must contain ten values",
    )
    deltas = []
    for index, expected in enumerate(actual["probabilities"]):
        received = _finite_number(
            probabilities[index], f"stress probe {name} probability must be finite"
        )
        _require(0 <= received <= 1, f"stress probe {name} probability is out of range")
        deltas.append(abs(received - expected))
    maximum = max(deltas, default=0.0)
    _require(maximum <= _TOLERANCE, f"stress probe {name} probability mismatch")
    return maximum


def verify_evaluation(model: Model, evaluation: Any, split: Any, test_cases: Any) -> dict[str, Any]:
    """Recompute and cross-check frozen report data using only bundled inference."""
    _require(isinstance(evaluation, dict), "evaluation report must be an object")
    info = model.info()
    training = info.get("training")
    _require(isinstance(training, dict), "model training metadata is missing")
    model_id = info.get("model_id")
    _require(
        isinstance(model_id, str) and _HASH.fullmatch(model_id) is not None,
        "model ID must be a lowercase SHA256 digest",
    )
    _require(
        evaluation.get("model_sha256") == model_id,
        "evaluation model hash does not match the bundled model",
    )
    _require(
        evaluation.get("dataset_sha256") == training.get("dataset_sha256"),
        "evaluation dataset hash does not match model metadata",
    )
    _require(
        evaluation.get("train_indices_sha256") == training.get("train_indices_sha256"),
        "evaluation train-index hash does not match model metadata",
    )
    config = evaluation.get("config")
    _require(
        isinstance(config, dict) and _canonical_sha256(config) == _canonical_sha256(_CONFIG),
        "evaluation training configuration mismatch",
    )
    _require(evaluation.get("converged") is True, "training report does not confirm convergence")

    train, test = _validate_split(split)
    train_sha = _canonical_sha256(train)
    test_sha = _canonical_sha256(test)
    _require(train_sha == training.get("train_indices_sha256"), "train-index hash mismatch")
    _require(
        train_sha == evaluation.get("train_indices_sha256"), "evaluation train-index hash mismatch"
    )
    _require(test_sha == evaluation.get("test_indices_sha256"), "test-index hash mismatch")
    _require(
        _exact_int(evaluation.get("train_count"), "train_count must be an integer") == len(train),
        "train_count mismatch",
    )
    _require(
        _exact_int(evaluation.get("test_count"), "test_count must be an integer") == len(test),
        "test_count mismatch",
    )

    _require(
        isinstance(test_cases, list) and len(test_cases) == len(test),
        "test cases must match the complete held-out split",
    )
    _require(
        _canonical_sha256(test_cases) == evaluation.get("test_cases_sha256"),
        "test-cases hash mismatch",
    )
    labels = []
    actual_predictions = []
    actual_probabilities = []
    report_predictions = evaluation.get("predictions")
    _require(
        isinstance(report_predictions, list) and len(report_predictions) == len(test),
        "prediction rows must match test cases",
    )
    max_probability_delta = 0.0
    for position, (case, expected_id, report_prediction) in enumerate(
        zip(test_cases, test, report_predictions, strict=True)
    ):
        _require(
            isinstance(case, dict) and set(case) == {"id", "pixels", "label"},
            f"test case {position} has an invalid schema",
        )
        case_id = _exact_int(
            case["id"], "test case ID must be an integer", minimum=0, maximum=_DATASET_SIZE - 1
        )
        _require(case_id == expected_id, "test case IDs must match sorted test indexes")
        pixels = case["pixels"]
        _require(
            isinstance(pixels, list) and len(pixels) == _PIXEL_COUNT,
            f"test case {case_id} must contain 64 pixels",
        )
        for pixel in pixels:
            _exact_int(
                pixel,
                f"test case {case_id} pixels must be integers from 0 to 16",
                minimum=0,
                maximum=16,
            )
        label = _exact_int(
            case["label"], f"test case {case_id} label must be an integer", minimum=0, maximum=9
        )
        labels.append(label)

        _require(
            isinstance(report_prediction, dict)
            and set(report_prediction) == {"id", "label", "predicted", "probabilities"},
            f"prediction {case_id} has an invalid schema",
        )
        _require(
            _exact_int(report_prediction["id"], "prediction ID must be an integer") == case_id,
            "prediction IDs must match sorted test cases",
        )
        _require(
            _exact_int(
                report_prediction["label"],
                "prediction label must be an integer",
                minimum=0,
                maximum=9,
            )
            == label,
            f"prediction label mismatch for test case {case_id}",
        )
        report_digit = _exact_int(
            report_prediction["predicted"],
            "predicted digit must be an integer",
            minimum=0,
            maximum=9,
        )
        report_probabilities = report_prediction["probabilities"]
        _require(
            isinstance(report_probabilities, list) and len(report_probabilities) == _CLASS_COUNT,
            f"prediction {case_id} must have ten probabilities",
        )
        for probability in report_probabilities:
            probability_value = _finite_number(probability, "probability must be finite")
            _require(0 <= probability_value <= 1, "probability must be between zero and one")
        _require(
            abs(math.fsum(report_probabilities) - 1.0) <= _TOLERANCE,
            f"prediction {case_id} probabilities must sum to one",
        )

        actual = model.predict(pixels)
        _require(
            report_digit == actual["digit"], f"predicted digit mismatch for test case {case_id}"
        )
        deltas = [
            abs(float(received) - expected)
            for received, expected in zip(
                report_probabilities, actual["probabilities"], strict=True
            )
        ]
        case_delta = max(deltas)
        _require(case_delta <= _TOLERANCE, f"probability mismatch for test case {case_id}")
        max_probability_delta = max(max_probability_delta, case_delta)
        actual_predictions.append(actual["digit"])
        actual_probabilities.append(actual["probabilities"])

    computed = _metrics(labels, actual_predictions, actual_probabilities)
    metrics = evaluation.get("metrics")
    _require(
        isinstance(metrics, dict) and set(metrics) == {"accuracy", "macro_f1", "log_loss"},
        "evaluation metrics have an invalid schema",
    )
    for name in ("accuracy", "macro_f1", "log_loss"):
        _matches_float(metrics[name], computed[name], f"metric {name}")

    matrix = evaluation.get("confusion_matrix")
    _require(
        isinstance(matrix, list) and len(matrix) == _CLASS_COUNT,
        "confusion matrix must contain ten rows",
    )
    for row_number, (row, expected_row) in enumerate(
        zip(matrix, computed["confusion_matrix"], strict=True)
    ):
        _require(
            isinstance(row, list) and len(row) == _CLASS_COUNT,
            "confusion matrix rows must contain ten values",
        )
        for cell, expected_cell in zip(row, expected_row, strict=True):
            _require(
                type(cell) is int and cell >= 0 and cell == expected_cell,
                f"confusion matrix mismatch at row {row_number}",
            )
    per_digit = evaluation.get("per_digit")
    _require(
        isinstance(per_digit, list) and len(per_digit) == _CLASS_COUNT,
        "per-digit metrics must contain ten rows",
    )
    for digit, (reported, expected) in enumerate(
        zip(per_digit, computed["per_digit"], strict=True)
    ):
        _require(
            isinstance(reported, dict)
            and set(reported) == {"digit", "precision", "recall", "f1", "support"},
            f"per-digit row {digit} has an invalid schema",
        )
        _require(
            _exact_int(reported["digit"], "per-digit class must be an integer") == digit,
            "per-digit classes must be ordered 0 through 9",
        )
        for name in ("precision", "recall", "f1"):
            _matches_float(reported[name], expected[name], f"digit {digit} {name}")
        _require(
            _exact_int(reported["support"], "per-digit support must be an integer")
            == expected["support"],
            f"digit {digit} support mismatch",
        )

    misclassified = [
        case_id
        for case_id, label, prediction in zip(test, labels, actual_predictions, strict=True)
        if label != prediction
    ]
    _require(evaluation.get("misclassified_ids") == misclassified, "misclassified IDs mismatch")
    _require(
        _exact_int(evaluation.get("error_count"), "error_count must be an integer")
        == len(misclassified),
        "error_count mismatch",
    )

    counts = evaluation.get("train_class_counts")
    _require(
        isinstance(counts, list) and len(counts) == _CLASS_COUNT,
        "training class counts must contain ten values",
    )
    for count in counts:
        _exact_int(count, "training class counts must be nonnegative integers", minimum=0)
    _require(sum(counts) == len(train), "training class counts do not sum to train_count")
    majority = min(range(_CLASS_COUNT), key=lambda digit: (-counts[digit], digit))
    baseline = evaluation.get("baseline")
    _require(
        isinstance(baseline, dict)
        and set(baseline) == {"selection", "majority_digit", "accuracy", "macro_f1"},
        "baseline report has an invalid schema",
    )
    _require(
        baseline["selection"] == "training majority class, ties lowest digit",
        "baseline selection policy mismatch",
    )
    _require(
        _exact_int(
            baseline["majority_digit"], "baseline digit must be an integer", minimum=0, maximum=9
        )
        == majority,
        "baseline majority digit mismatch",
    )
    baseline_metrics = _metrics(
        labels, [majority] * len(labels), [[1.0 / _CLASS_COUNT] * _CLASS_COUNT for _ in labels]
    )
    _matches_float(baseline["accuracy"], baseline_metrics["accuracy"], "baseline accuracy")
    _matches_float(baseline["macro_f1"], baseline_metrics["macro_f1"], "baseline macro_f1")

    probes = evaluation.get("stress_probes")
    _require(
        isinstance(probes, list) and len(probes) == 3,
        "evaluation must contain the three fixed stress probes",
    )
    expected_probes = [
        ("all_zero", [0] * _PIXEL_COUNT),
        ("all_16", [16] * _PIXEL_COUNT),
        ("checkerboard", [16 * ((row + column) % 2) for row in range(8) for column in range(8)]),
    ]
    for probe, (name, pixels) in zip(probes, expected_probes, strict=True):
        max_probability_delta = max(max_probability_delta, _check_probe(model, probe, name, pixels))

    return {
        "verified": True,
        "model_id": model_id,
        "train_count": len(train),
        "test_count": len(test),
        "accuracy": computed["accuracy"],
        "macro_f1": computed["macro_f1"],
        "log_loss": computed["log_loss"],
        "baseline_accuracy": baseline_metrics["accuracy"],
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "error_count": len(misclassified),
        "max_probability_delta": max_probability_delta,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify PixelProof's frozen held-out evaluation")
    parser.add_argument("--evaluation", type=Path, default=_ROOT / "docs" / "evaluation.json")
    parser.add_argument("--test-cases", type=Path, default=_ROOT / "data" / "test_cases.json")
    parser.add_argument("--split", type=Path, default=_ROOT / "data" / "split.json")
    args = parser.parse_args(argv)
    try:
        model = Model.bundled()
        result = verify_evaluation(
            model,
            _read_json(args.evaluation),
            _read_json(args.split),
            _read_json(args.test_cases),
        )
    except (OSError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
