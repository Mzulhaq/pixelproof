"""Local command-line prediction and loopback serving."""

import argparse
import json
import sys
from pathlib import Path

from pixelproof.model import MAX_REQUEST_BYTES, Model, strict_json, validate_payload


def _port(value):
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect and serve the frozen PixelProof model")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="Print model identity and training metadata")
    predict = commands.add_parser("predict", help="Predict a JSON file containing images")
    predict.add_argument("path", type=Path)
    serve = commands.add_parser("serve", help="Serve the inference API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=_port)
    args = parser.parse_args(argv)
    try:
        model = Model.bundled()
        if args.command == "info":
            result = model.info()
        elif args.command == "predict":
            with args.path.open("rb") as handle:
                raw = handle.read(MAX_REQUEST_BYTES + 1)
            if len(raw) > MAX_REQUEST_BYTES:
                raise ValueError("Input exceeds 32 KiB")
            images = validate_payload(strict_json(raw))
            result = {"model_id": model.model_id, "predictions": model.predict_batch(images)}
        else:
            import uvicorn

            from pixelproof.api import create_app

            uvicorn.run(create_app(model), host=args.host, port=args.port, access_log=False)
            return 0
        print(json.dumps(result, allow_nan=False))
        return 0
    except (ValueError, OSError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
