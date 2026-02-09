FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY streamlit_app.py /app/streamlit_app.py

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir .[streamlit]

ENV PORT=8080
EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
