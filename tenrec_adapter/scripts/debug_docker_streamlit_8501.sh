#!/bin/bash
set -euo pipefail

# Diagnose docker port mapping and Streamlit service.
#
# Usage:
#   bash tenrec_adapter/scripts/debug_docker_streamlit_8501.sh
#   CONTAINER=zhl CONTAINER_PORT=8501 HOST_PORT=8501 bash tenrec_adapter/scripts/debug_docker_streamlit_8501.sh

CONTAINER="${CONTAINER:-zhl}"
CONTAINER_PORT="${CONTAINER_PORT:-8501}"
HOST_PORT="${HOST_PORT:-${CONTAINER_PORT}}"
APP_PATH="${APP_PATH:-tenrec_adapter/app.py}"

echo "=== Docker Container ==="
docker ps -a --filter "name=${CONTAINER}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo

echo "=== docker port ==="
docker port "${CONTAINER}" || true
echo

echo "=== Host listen on :${HOST_PORT} ==="
ss -lntp | grep ":${HOST_PORT}" || echo "No process is listening on :${HOST_PORT} on host."
echo

echo "=== In-container listen on :${CONTAINER_PORT} ==="
docker exec "${CONTAINER}" bash -lc "ss -lntp | grep ':${CONTAINER_PORT}' || true"
echo

echo "=== In-container streamlit process ==="
docker exec "${CONTAINER}" bash -lc "ps -ef | grep 'streamlit run ${APP_PATH}' | grep -v grep || true"
docker exec "${CONTAINER}" bash -lc "ps -ef | grep 'python3 -m streamlit run ${APP_PATH}' | grep -v grep || true"
echo

echo "=== In-container streamlit command/module check ==="
docker exec "${CONTAINER}" bash -lc "command -v streamlit || echo 'streamlit command not found'"
docker exec "${CONTAINER}" bash -lc "python3 - <<'PY'
try:
    import streamlit
    print('streamlit module OK:', streamlit.__version__)
except Exception as e:
    print('streamlit module FAIL:', e)
PY"
echo

echo "=== In-container HTTP check ==="
docker exec "${CONTAINER}" bash -lc "python3 - <<'PY'
import urllib.request
url = 'http://127.0.0.1:${CONTAINER_PORT}'
try:
    with urllib.request.urlopen(url, timeout=5) as r:
        print('OK', r.status)
except Exception as e:
    print('FAIL', e)
PY"
echo

echo "=== Host HTTP check ==="
python3 - <<PY
import urllib.request
url = 'http://127.0.0.1:${HOST_PORT}'
try:
    with urllib.request.urlopen(url, timeout=5) as r:
        print('OK', r.status)
except Exception as e:
    print('FAIL', e)
PY
echo

echo "=== Streamlit logs (latest 120 lines) ==="
docker exec "${CONTAINER}" bash -lc "ls -1t /root/zhl/x-algorithm/logs/streamlit/streamlit_*.log 2>/dev/null | head -n 1 | xargs -r tail -n 120"
echo

echo "If host is behind Nginx and you still see HTTP 502, check upstream config:"
echo "  proxy_pass http://127.0.0.1:${HOST_PORT};"
