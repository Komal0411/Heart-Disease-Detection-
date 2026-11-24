#!/usr/bin/env bash
# Start the API + frontend on http://127.0.0.1:8000
set -e
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
