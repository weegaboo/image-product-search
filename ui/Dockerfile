FROM python:3.12-slim

WORKDIR /app

RUN pip install --upgrade pip && pip install uv

COPY pyproject.toml uv.lock ./
COPY ui ./ui

RUN uv pip install --system --no-cache .

CMD ["streamlit", "run", "ui/application.py", "--server.port=8501", "--server.address=0.0.0.0"]