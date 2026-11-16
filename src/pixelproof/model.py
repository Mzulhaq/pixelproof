"""Strict JSON model loading and immutable, dependency-free inference."""

import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass
from importlib.resources import files

MAX_MODEL_BYTES = 256 * 1024
MAX_REQUEST_BYTES = 32 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_KEYS = {
    "schema_version",
    "feature_count",
    "classes",
    "pixel_max",
    "preprocessing",
    "coef",
    "intercept",
    "training",
}
_TRAINING_FIXED = {
    "dataset": "sklearn.load_digits",
    "seed": 42,
    "test_fraction": 0.25,
    "estimator": "LogisticRegression",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "tol": 1e-4,
}
_TRAINING_EXTRA = {"dataset_sha256", "train_indices_sha256", "sklearn_version", "numpy_version"}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("Nonfinite JSON number")


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite JSON number")
    return result


def strict_json(data: bytes):
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique,
            parse_constant=_nonfinite,
            parse_float=_float,
        )
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid JSON encoding or nesting") from error


def _number(value):
    if type(value) not in (int, float) or not -100 <= value <= 100 or not math.isfinite(value):
        raise ValueError("Coefficients must be finite numbers with magnitude at most 100")
    return float(value)


def validate_pixels(pixels) -> tuple[int, ...]:
    if not isinstance(pixels, list) or len(pixels) != 64:
        raise ValueError("Each image must contain exactly 64 integer pixels")
    values = tuple(pixels)
    if any(type(pixel) is not int or not 0 <= pixel <= 16 for pixel in values):
        raise ValueError("Pixels must be integers from 0 through 16; booleans are invalid")
    return values


def validate_payload(payload) -> list[list[int]]:
    if not isinstance(payload, dict) or set(payload) != {"images"}:
        raise ValueError("Payload requires exactly the images field")
    images = payload["images"]
    if not isinstance(images, list) or not 1 <= len(images) <= 32:
        raise ValueError("Batch must contain 1-32 images")
    return [list(validate_pixels(pixels)) for pixels in images]


@dataclass(frozen=True)
class Model:
    model_id: str
    _coef: tuple[tuple[float, ...], ...]
    _intercept: tuple[float, ...]
    _training: tuple[tuple[str, str | int | float], ...]

    @classmethod
    def from_bytes(cls, data: bytes, expected_sha256: str | None = None):
        if not isinstance(data, bytes) or len(data) > MAX_MODEL_BYTES:
            raise ValueError("Model must be bytes with size at most 256 KiB")
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str)
            or not _SHA.fullmatch(expected_sha256)
            or not hmac.compare_digest(digest, expected_sha256)
        ):
            raise ValueError("Model SHA256 does not match the expected digest")
        obj = strict_json(data)
        if not isinstance(obj, dict) or set(obj) != _MODEL_KEYS:
            raise ValueError("Model fields do not match schema")
        for field, expected in (("schema_version", 1), ("feature_count", 64), ("pixel_max", 16)):
            if type(obj[field]) is not int or obj[field] != expected:
                raise ValueError(f"Unsupported {field}")
        if (
            obj["classes"] != list(range(10))
            or not isinstance(obj["classes"], list)
            or any(type(c) is not int for c in obj["classes"])
        ):
            raise ValueError("Classes must be integers 0-9 in order")
        if obj["preprocessing"] != "divide_by_16":
            raise ValueError("Unsupported preprocessing")
        coef, intercept = obj["coef"], obj["intercept"]
        if not isinstance(coef, list) or len(coef) != 10:
            raise ValueError("Model needs ten coefficient rows")
        if any(not isinstance(row, list) or len(row) != 64 for row in coef):
            raise ValueError("Coefficient rows must have 64 values")
        if not isinstance(intercept, list) or len(intercept) != 10:
            raise ValueError("Model needs ten intercepts")
        training = obj["training"]
        if (
            not isinstance(training, dict)
            or set(training) != set(_TRAINING_FIXED) | _TRAINING_EXTRA
        ):
            raise ValueError("Training metadata does not match schema")
        for field, expected in _TRAINING_FIXED.items():
            value = training[field]
            allowed_types = (int, float) if type(expected) is float else (type(expected),)
            if type(value) not in allowed_types or value != expected:
                raise ValueError(f"Unsupported training {field}")
        for field in ("dataset_sha256", "train_indices_sha256"):
            if not isinstance(training[field], str) or not _SHA.fullmatch(training[field]):
                raise ValueError("Invalid training SHA256")
        for field in ("sklearn_version", "numpy_version"):
            version = training[field]
            if (
                not isinstance(version, str)
                or not 1 <= len(version) <= 64
                or any(not 32 <= ord(char) <= 126 for char in version)
            ):
                raise ValueError("Invalid training version")
        return cls(
            digest,
            tuple(tuple(_number(v) for v in row) for row in coef),
            tuple(_number(v) for v in intercept),
            tuple(sorted(training.items())),
        )

    @classmethod
    def bundled(cls):
        directory = files("pixelproof").joinpath("assets")
        with directory.joinpath("model.json").open("rb") as handle:
            data = handle.read(MAX_MODEL_BYTES + 1)
        expected = directory.joinpath("model.sha256").read_text(encoding="ascii").strip()
        return cls.from_bytes(data, expected)

    def info(self):
        return {
            "model_id": self.model_id,
            "schema_version": 1,
            "feature_count": 64,
            "classes": list(range(10)),
            "pixel_max": 16,
            "preprocessing": "divide_by_16",
            "training": dict(self._training),
        }

    def _predict(self, pixels: tuple[int, ...]) -> dict:
        normalized = tuple(pixel / 16 for pixel in pixels)
        logits = [
            math.fsum(weight * value for weight, value in zip(row, normalized, strict=True)) + bias
            for row, bias in zip(self._coef, self._intercept, strict=True)
        ]
        maximum = max(logits)
        exponentials = [math.exp(logit - maximum) for logit in logits]
        denominator = math.fsum(exponentials)
        probabilities = [value / denominator for value in exponentials]
        return {
            "digit": max(range(10), key=lambda digit: probabilities[digit]),
            "probabilities": probabilities,
        }

    def predict(self, pixels: list[int]) -> dict:
        return self._predict(validate_pixels(pixels))

    def predict_batch(self, images: list[list[int]]) -> list[dict]:
        validated = validate_payload({"images": images})
        return [self._predict(tuple(pixels)) for pixels in validated]
