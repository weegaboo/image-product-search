FROM python:3.12-slim

WORKDIR /app

RUN pip install --upgrade pip && pip install uv

COPY pyproject.toml uv.lock ./
COPY ./search/application.py .

RUN uv pip install --system --no-cache .

CMD [ \
    "uvicorn", \
    "application:app", \
    "--host", \
    "0.0.0.0", \
    "--port", \
    "8000", \
    "--loop", \
    "uvloop", \
    "--http", \
    "httptools", \
    "--timeout-keep-alive", \
    "90", \
    "--timeout-graceful-shutdown", \
    "10" \
]
