"""Exercise actual loopback HTTP with attributed holdout pixels and a synthetic blank."""

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn

from pixelproof.api import create_app
from pixelproof.model import Model

ROOT = Path(__file__).resolve().parents[1]


def run_demo():
    model = Model.bundled()
    cases = json.loads((ROOT / "data" / "test_cases.json").read_text(encoding="utf-8"))[:3]
    images = [case["pixels"] for case in cases] + [[0] * 64]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(model), log_level="error", access_log=False))
    worker = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with opener.open(req, timeout=5) as response:
            return json.load(response)

    try:
        worker.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if not worker.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("Demo server did not start")
            time.sleep(0.01)
        assert request("/health") == {"status": "ok", "model_id": model.model_id}
        assert request("/v1/model") == model.info()
        result = request("/v1/predict", {"images": images})
        assert result == {"model_id": model.model_id, "predictions": model.predict_batch(images)}
        try:
            request("/v1/predict", {"images": [[17] * 64]})
        except urllib.error.HTTPError as error:
            assert error.code == 422
        else:
            raise AssertionError("Invalid pixel request was accepted")
        return {
            "model_id": model.model_id,
            "http_checks": "passed",
            "holdout_examples": [
                {"id": case["id"], "label": case["label"], "predicted": pred["digit"]}
                for case, pred in zip(cases, result["predictions"][:3], strict=True)
            ],
            "synthetic_blank": result["predictions"][-1],
            "note": "These examples are not an accuracy estimate; run scripts/evaluate.py.",
        }
    finally:
        server.should_exit = True
        if worker.ident is not None:
            worker.join(timeout=10)
        sock.close()
        if worker.is_alive():
            raise RuntimeError("Demo server failed to stop")


if __name__ == "__main__":
    print(json.dumps(run_demo(), sort_keys=True, allow_nan=False))
