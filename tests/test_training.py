from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
load_digits = importlib.import_module("sklearn.datasets").load_digits
training = importlib.import_module("pixelproof.training")
build_bundle = training.build_bundle
write_bundle = training.write_bundle
Model = importlib.import_module("pixelproof.model").Model


@pytest.fixture(scope="module")
def bundle():
    return build_bundle()


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def test_build_bundle_has_exact_artifact_contract_and_fixed_training_metadata(bundle):
    artifact = bundle["model"]
    assert set(bundle) == {"model", "evaluation", "split", "test_cases"}
    assert set(artifact) == {
        "schema_version",
        "feature_count",
        "classes",
        "pixel_max",
        "preprocessing",
        "coef",
        "intercept",
        "training",
    }
    assert artifact["schema_version"] == 1
    assert artifact["feature_count"] == 64
    assert artifact["classes"] == list(range(10))
    assert artifact["pixel_max"] == 16
    assert artifact["preprocessing"] == "divide_by_16"
    assert len(artifact["coef"]) == 10 and all(len(row) == 64 for row in artifact["coef"])
    assert len(artifact["intercept"]) == 10
    assert set(artifact["training"]) == {
        "dataset",
        "dataset_sha256",
        "train_indices_sha256",
        "seed",
        "test_fraction",
        "estimator",
        "C",
        "solver",
        "max_iter",
        "tol",
        "sklearn_version",
        "numpy_version",
    }
    training = artifact["training"]
    assert training["dataset"] == "sklearn.load_digits"
    assert training["seed"] == 42 and training["test_fraction"] == 0.25
    assert (training["estimator"], training["C"], training["solver"]) == (
        "LogisticRegression",
        1.0,
        "lbfgs",
    )
    assert training["max_iter"] == 1000 and training["tol"] == 1e-4
    for field in ("dataset_sha256", "train_indices_sha256"):
        assert len(training[field]) == 64
        assert set(training[field]) <= set("0123456789abcdef")
    for version in (training["sklearn_version"], training["numpy_version"]):
        assert 1 <= len(version) <= 64 and all(32 <= ord(char) <= 126 for char in version)
    assert bundle["evaluation"]["converged"] is True


def test_split_is_sorted_disjoint_stratified_and_uses_all_digits(bundle):
    train = bundle["split"]["train"]
    test = bundle["split"]["test"]
    assert train == sorted(train) and test == sorted(test)
    assert not set(train) & set(test)
    assert sorted(train + test) == list(range(1797))
    assert len(test) == 450
    labels = load_digits().target
    assert set(labels[test]) == set(range(10))
    assert set(labels[train]) == set(range(10))
    assert bundle["evaluation"]["train_count"] == len(train)
    assert bundle["evaluation"]["test_count"] == len(test)


def test_dataset_and_train_index_hashes_cover_canonical_integer_data(bundle):
    dataset = load_digits()
    payload = {
        "pixels": dataset.data.astype(int).tolist(),
        "labels": dataset.target.astype(int).tolist(),
    }
    expected_dataset_hash = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    expected_train_hash = hashlib.sha256(canonical_bytes(bundle["split"]["train"])).hexdigest()
    expected_test_hash = hashlib.sha256(canonical_bytes(bundle["split"]["test"])).hexdigest()
    expected_cases_hash = hashlib.sha256(canonical_bytes(bundle["test_cases"])).hexdigest()
    metadata = bundle["model"]["training"]
    assert metadata["dataset_sha256"] == expected_dataset_hash
    assert metadata["train_indices_sha256"] == expected_train_hash
    assert bundle["evaluation"]["dataset_sha256"] == expected_dataset_hash
    assert bundle["evaluation"]["test_indices_sha256"] == expected_test_hash
    assert bundle["evaluation"]["test_cases_sha256"] == expected_cases_hash


def test_exported_runtime_matches_sklearn_probabilities_on_all_heldout_cases(bundle):
    artifact_bytes = canonical_bytes(bundle["model"])
    runtime_model = Model.from_bytes(artifact_bytes, bundle["evaluation"]["model_sha256"])
    for case, expected in zip(
        bundle["test_cases"], bundle["evaluation"]["predictions"], strict=True
    ):
        actual = runtime_model.predict(case["pixels"])
        assert actual["digit"] == expected["predicted"]
        assert actual["probabilities"] == pytest.approx(expected["probabilities"], abs=1e-10, rel=0)


def test_baseline_majority_uses_training_labels_only(bundle):
    dataset = load_digits()
    train_labels = dataset.target[bundle["split"]["train"]]
    test_labels = dataset.target[bundle["split"]["test"]]
    counts = Counter(int(label) for label in train_labels)
    majority = min(range(10), key=lambda label: (-counts[label], label))
    baseline = bundle["evaluation"]["baseline"]
    assert baseline["selection"] == "training majority class, ties lowest digit"
    assert baseline["majority_digit"] == majority
    assert bundle["evaluation"]["train_class_counts"] == [counts[label] for label in range(10)]
    assert baseline["accuracy"] == pytest.approx(float(np.mean(test_labels == majority)))
    assert "log_loss" not in baseline
    assert "macro_f1" in baseline


def test_heldout_metrics_errors_and_probabilities_are_complete_and_valid(bundle):
    evaluation = bundle["evaluation"]
    test_cases = bundle["test_cases"]
    assert len(test_cases) == 450
    assert [case["id"] for case in test_cases] == bundle["split"]["test"]
    assert all(len(case["pixels"]) == 64 for case in test_cases)
    assert all(
        all(isinstance(value, int) and 0 <= value <= 16 for value in case["pixels"])
        for case in test_cases
    )
    dataset = load_digits()
    assert all(
        case["pixels"] == dataset.data[case["id"]].astype(int).tolist() for case in test_cases
    )
    assert all(case["label"] == int(dataset.target[case["id"]]) for case in test_cases)
    assert set(evaluation["metrics"]) == {"accuracy", "macro_f1", "log_loss"}
    assert 0 <= evaluation["metrics"]["accuracy"] <= 1
    assert 0 <= evaluation["metrics"]["macro_f1"] <= 1
    assert evaluation["metrics"]["log_loss"] >= 0
    assert len(evaluation["per_digit"]) == 10
    assert [row["digit"] for row in evaluation["per_digit"]] == list(range(10))
    assert len(evaluation["confusion_matrix"]) == 10
    assert all(len(row) == 10 for row in evaluation["confusion_matrix"])
    assert sum(map(sum, evaluation["confusion_matrix"])) == len(test_cases)
    predictions = evaluation["predictions"]
    assert len(predictions) == len(test_cases)
    assert [prediction["id"] for prediction in predictions] == [case["id"] for case in test_cases]
    for prediction in predictions:
        probabilities = prediction["probabilities"]
        assert prediction["predicted"] in range(10)
        assert len(probabilities) == 10
        assert all(np.isfinite(value) and 0 <= value <= 1 for value in probabilities)
        assert sum(probabilities) == pytest.approx(1.0, abs=1e-12)
    assert evaluation["misclassified_ids"] == [
        row["id"] for row in predictions if row["predicted"] != row["label"]
    ]
    assert len(evaluation["stress_probes"]) == 3
    assert {probe["name"] for probe in evaluation["stress_probes"]} == {
        "all_zero",
        "all_16",
        "checkerboard",
    }
    assert all(
        "label" not in probe and "accuracy" not in probe for probe in evaluation["stress_probes"]
    )


def test_build_bundle_reproduces_split_cases_and_metadata():
    first = build_bundle()
    second = build_bundle()
    assert first["split"] == second["split"]
    assert first["test_cases"] == second["test_cases"]
    assert first["model"]["training"] == second["model"]["training"]


def test_committed_holdout_and_baseline_match_source_dataset(bundle):
    root = Path(__file__).resolve().parents[1]

    def read(path):
        return json.loads((root / path).read_text(encoding="utf-8"))

    assert read("data/split.json") == bundle["split"]
    assert read("data/test_cases.json") == bundle["test_cases"]
    recorded = read("docs/evaluation.json")
    for key in ("dataset_sha256", "train_class_counts", "test_cases_sha256"):
        assert recorded[key] == bundle["evaluation"][key]
    assert Model.bundled().info()["training"]["dataset_sha256"] == recorded["dataset_sha256"]


def test_write_bundle_creates_five_files_with_exact_model_digest(tmp_path, bundle):
    output = tmp_path / "candidate"
    write_bundle(output)
    assert set(path.name for path in output.iterdir()) == {
        "model.json",
        "model.sha256",
        "evaluation.json",
        "split.json",
        "test_cases.json",
    }
    model_bytes = (output / "model.json").read_bytes()
    model = json.loads(model_bytes)
    assert model == bundle["model"]
    assert model_bytes == canonical_bytes(model)
    digest = hashlib.sha256(model_bytes).hexdigest()
    assert (output / "model.sha256").read_text(encoding="ascii") == digest + "\n"
    assert json.loads((output / "evaluation.json").read_text()) == bundle["evaluation"]
    assert json.loads((output / "split.json").read_text()) == bundle["split"]
    assert json.loads((output / "test_cases.json").read_text()) == bundle["test_cases"]
    assert all(
        (output / name).read_bytes().endswith(b"\n")
        for name in ("evaluation.json", "split.json", "test_cases.json")
    )


def test_write_bundle_refuses_existing_output_and_preserves_it(tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not overwrite")
    with pytest.raises(FileExistsError):
        write_bundle(output)
    assert sentinel.read_text() == "do not overwrite"


def test_write_bundle_requires_output_path(tmp_path):
    with pytest.raises((TypeError, ValueError)):
        write_bundle(None)


def test_train_script_writes_bundle_and_prints_actual_summary(tmp_path):
    output = tmp_path / "from-script"
    project_root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "train.py"), str(output)],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    summary = json.loads(process.stdout)
    assert summary["test_count"] == 450
    assert (output / "model.json").exists()
