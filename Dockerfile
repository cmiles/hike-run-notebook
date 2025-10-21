# syntax=docker/dockerfile:1.7
FROM python:3.13.7-slim

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- deps (cache-friendly) ---
COPY --link requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- app files ---
# This brings in hike-run-notebook.py and date_tools.py (good)
COPY --link . .

# Railway provides $PORT; use 8080 locally
ENV PORT=8080

# Non-root user + ensure write perms in /app
RUN useradd -m app_user && chown -R app_user:app_user /app
USER app_user

EXPOSE 8080

# App mode + hide code
CMD marimo run --host 0.0.0.0 --port $PORT hike-run-notebook.py

