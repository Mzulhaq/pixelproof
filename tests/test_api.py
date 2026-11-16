import json

import pytest
from fastapi.testclient import TestClient

from pixelproof.api import create_app
from pixelproof.model import Model
from tests.test_model import encoded


@pytest.fixture
def client():
    with TestClient(create_app(Model.from_bytes(encoded()))) as client:
        yield client


def test_prediction_identity_metadata_and_batch(client):
    metadata = client.get("/v1/model").json()
    assert "coef" not in metadata
    assert client.get("/health").json() == {"status": "ok", "model_id": metadata["model_id"]}
    response = client.post("/v1/predict", json={"images": [[0] * 64, [16] * 64]})
    assert response.status_code == 200
    assert response.json() == {
        "model_id": metadata["model_id"],
        "predictions": [
            {"digit": 0, "probabilities": [0.1] * 10},
            {"digit": 0, "probabilities": [0.1] * 10},
        ],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"images": []},
        {"images": [[1.0] * 64]},
        {"images": [[False] * 64]},
        {"images": [[0] * 63]},
        {"images": [[0] * 64], "debug": True},
        {"images": [[0] * 64] * 33},
        {"images": [[0] * 64, [99] * 64]},
    ],
)
def test_bad_schema_and_pixels_return_422(client, payload):
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422
    assert "predictions" not in response.json()


@pytest.mark.parametrize(
    "raw",
    [
        b'{"images":[],"images":[]}',
        b'{"images":NaN}',
        b'{"images":1e999}',
        b"{",
        b"\xff",
        b"[" * 2000,
    ],
)
def test_invalid_json_returns_400_without_echo(client, raw):
    response = client.post("/v1/predict", content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid JSON"}


def test_wrong_media_type_and_declared_oversize(client):
    assert client.post("/v1/predict", content="{}").status_code == 415
    response = client.post(
        "/v1/predict", content=b"x" * (32 * 1024 + 1), headers={"content-type": "application/json"}
    )
    assert response.status_code == 413


def test_streamed_oversize_is_rejected_without_content_length(client):
    def chunks():
        yield b" " * (32 * 1024)
        yield json.dumps({"images": [[0] * 64]}).encode()

    response = client.post(
        "/v1/predict", content=chunks(), headers={"content-type": "application/json"}
    )
    assert "content-length" not in response.request.headers
    assert response.status_code == 413


def test_misleading_small_content_length_does_not_bypass_body_limit(client):
    response = client.post(
        "/v1/predict",
        content=b" " * (32 * 1024 + 1),
        headers={"content-type": "application/json", "content-length": "1"},
    )
    assert response.status_code == 413
