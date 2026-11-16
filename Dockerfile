FROM python:3.13-slim
WORKDIR /app
COPY requirements-runtime.lock .
RUN pip install --no-cache-dir -r requirements-runtime.lock
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY scripts ./scripts
COPY data ./data
COPY docs/evaluation.json ./docs/evaluation.json
USER 10001:10001
CMD ["python", "scripts/demo.py"]
