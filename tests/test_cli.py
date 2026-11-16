import json
import subprocess
import sys

import pytest

from pixelproof.cli import main
from pixelproof.model import Model
from tests.test_model import encoded


def test_cli_prediction_and_info(tmp_path, capsys, monkeypatch):
    model = Model.from_bytes(encoded())
    monkeypatch.setattr(Model, "bundled", classmethod(lambda cls: model))
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"images": [[0] * 64]}), encoding="utf-8")
    assert main(["predict", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["predictions"][0]["digit"] == 0
    assert main(["info"]) == 0
    assert json.loads(capsys.readouterr().out)["model_id"] == model.model_id


@pytest.mark.parametrize("data", [b"{}", b'{"images":NaN}', b"x" * (32 * 1024 + 1)])
def test_cli_bad_input_is_error(tmp_path, capsys, monkeypatch, data):
    monkeypatch.setattr(Model, "bundled", classmethod(lambda cls: Model.from_bytes(encoded())))
    path = tmp_path / "bad.json"
    path.write_bytes(data)
    assert main(["predict", str(path)]) == 1
    assert json.loads(capsys.readouterr().err)["error"]


def test_installed_module_help():
    result = subprocess.run(
        [sys.executable, "-m", "pixelproof.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "predict" in result.stdout
