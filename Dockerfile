FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY backend /app/backend
COPY frontend /app/frontend

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir .[streamlit,models]

ENV PORT=8080
EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "frontend/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
