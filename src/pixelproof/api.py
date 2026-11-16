"""A bounded, local inference API. No submitted pixels are stored or logged."""

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from pixelproof.model import MAX_REQUEST_BYTES, Model, strict_json, validate_payload


def create_app(model: Model | None = None) -> FastAPI:
    loaded = model if model is not None else Model.bundled()
    app = FastAPI(title="PixelProof", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_id": loaded.model_id}

    @app.get("/v1/model")
    def model_info():
        return loaded.info()

    @app.post("/v1/predict")
    async def predict(request: Request):
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != (
            "application/json"
        ):
            raise HTTPException(415, "Content-Type must be application/json")
        body = bytearray()
        try:
            async for chunk in request.stream():
                if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                    raise HTTPException(413, "Request body exceeds 32 KiB")
                body.extend(chunk)
        except ClientDisconnect as error:
            raise HTTPException(400, "Incomplete request body") from error
        try:
            payload = strict_json(bytes(body))
        except ValueError as error:
            raise HTTPException(400, "Invalid JSON") from error
        try:
            images = validate_payload(payload)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        predictions = await run_in_threadpool(loaded.predict_batch, images)
        return {"model_id": loaded.model_id, "predictions": predictions}

    return app
