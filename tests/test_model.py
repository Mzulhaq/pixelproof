import hashlib
import json
import math

import pytest

from pixelproof.model import Model


def artifact():
    return {
        "schema_version": 1,
        "feature_count": 64,
        "classes": list(range(10)),
        "pixel_max": 16,
        "preprocessing": "divide_by_16",
        "coef": [[0.0] * 64 for _ in range(10)],
        "intercept": [0.0] * 10,
        "training": {
            "dataset": "sklearn.load_digits",
            "dataset_sha256": "a" * 64,
            "train_indices_sha256": "b" * 64,
            "seed": 42,
            "test_fraction": 0.25,
            "estimator": "LogisticRegression",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "tol": 1e-4,
            "sklearn_version": "1.9.1",
            "numpy_version": "2.4.0",
        },
    }


def encoded(value=None):
    return json.dumps(artifact() if value is None else value).encode()


def test_uniform_probabilities_lowest_digit_tie_and_artifact_identity():
    raw = encoded()
    model = Model.from_bytes(raw, hashlib.sha256(raw).hexdigest())
    assert model.model_id == hashlib.sha256(raw).hexdigest()
    assert model.predict([0] * 64) == {"digit": 0, "probabilities": [0.1] * 10}


def test_input_scaling_and_softmax_match_hand_computed_oracle():
    payload = artifact()
    payload["coef"][3][0] = 1.0
    model = Model.from_bytes(encoded(payload))
    result = model.predict([16] + [0] * 63)
    assert result["digit"] == 3
    assert result["probabilities"][3] == pytest.approx(math.e / (math.e + 9))
    assert sum(result["probabilities"]) == pytest.approx(1)


def test_extreme_valid_logits_are_numerically_stable():
    payload = artifact()
    payload["coef"] = [[100.0 if row == 9 else -100.0] * 64 for row in range(10)]
    result = Model.from_bytes(encoded(payload)).predict([16] * 64)
    assert result["digit"] == 9
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in result["probabilities"])
    assert sum(result["probabilities"]) == pytest.approx(1)


@pytest.mark.parametrize(
    "pixels",
    [
        [],
        [0] * 63,
        [0] * 65,
        [False] * 64,
        [1.0] * 64,
        ["1"] * 64,
        [-1] * 64,
        [17] * 64,
        None,
        [[0] * 8] * 8,
    ],
)
def test_invalid_pixels_rejected(pixels):
    with pytest.raises(ValueError):
        Model.from_bytes(encoded()).predict(pixels)


def test_batch_bounds_and_invalid_member_rejected():
    model = Model.from_bytes(encoded())
    assert len(model.predict_batch([[0] * 64] * 32)) == 32
    for images in ([], [[0] * 64] * 33, [[0] * 64, [99] * 64], None):
        with pytest.raises(ValueError):
            model.predict_batch(images)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(schema_version=True),
        lambda d: d.update(feature_count=63),
        lambda d: d.update(classes=[False] + list(range(1, 10))),
        lambda d: d.update(pixel_max=255),
        lambda d: d.update(preprocessing="exec"),
        lambda d: d.update(coef=[[0] * 63] * 10),
        lambda d: d.update(intercept=[0] * 9),
        lambda d: d["coef"][0].__setitem__(0, float("nan")),
        lambda d: d["coef"][0].__setitem__(0, 101),
        lambda d: d["coef"][0].__setitem__(0, True),
        lambda d: d.update(extra="unexpected"),
        lambda d: d["training"].update(seed=False),
        lambda d: d["training"].update(C=2),
        lambda d: d["training"].update(dataset_sha256="bad"),
        lambda d: d["training"].update(sklearn_version="bad\nversion"),
    ],
)
def test_malformed_artifact_fails_closed(change):
    payload = artifact()
    change(payload)
    with pytest.raises(ValueError):
        Model.from_bytes(encoded(payload))


def test_digest_mismatch_duplicate_keys_and_size_limit_rejected():
    with pytest.raises(ValueError, match="SHA256"):
        Model.from_bytes(encoded(), "0" * 64)
    for data in (b'{"schema_version":1,"schema_version":1}', b"x" * (256 * 1024 + 1), b"\xff"):
        with pytest.raises(ValueError):
            Model.from_bytes(data)


def test_metadata_does_not_mutate_runtime_and_has_no_weights():
    model = Model.from_bytes(encoded())
    metadata = model.info()
    assert "coef" not in metadata and "intercept" not in metadata
    metadata["classes"][0] = 9
    metadata["training"]["seed"] = 999
    assert model.info()["classes"] == list(range(10))
    assert model.info()["training"]["seed"] == 42
    with pytest.raises(AttributeError):
        model.model_id = "changed"
