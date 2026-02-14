#!/bin/bash
set -euo pipefail

# Start Streamlit inside docker container and run health checks.
#
# Usage:
#   bash tenrec_adapter/scripts/start_streamlit_in_docker.sh
#   CONTAINER=zhl CONTAINER_PORT=8501 HOST_PORT=8501 bash tenrec_adapter/scripts/start_streamlit_in_docker.sh

CONTAINER="${CONTAINER:-zhl}"
CONTAINER_PORT="${CONTAINER_PORT:-8501}"
HOST_PORT="${HOST_PORT:-${CONTAINER_PORT}}"
APP_PATH="${APP_PATH:-tenrec_adapter/app.py}"
WORKDIR="${WORKDIR:-/root/zhl/x-algorithm}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/streamlit}"

echo "[1/5] Check container status"
docker ps --filter "name=${CONTAINER}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo "[2/5] Stop old streamlit process in container (if exists)"
docker exec "${CONTAINER}" bash -lc "
  pids=\$(pgrep -f '^streamlit run ' || true)
  if [ -n \"\${pids}\" ]; then kill \${pids} || true; fi
  pids=\$(pgrep -f '^python3 -m streamlit run ' || true)
  if [ -n \"\${pids}\" ]; then kill \${pids} || true; fi
"

echo "[3/5] Start streamlit"
docker exec -d "${CONTAINER}" bash -lc "
  cd ${WORKDIR} &&
  mkdir -p ${LOG_DIR} &&
  if command -v streamlit >/dev/null 2>&1; then
    STREAMLIT_CMD='streamlit'
  else
    STREAMLIT_CMD='python3 -m streamlit'
  fi &&
  \${STREAMLIT_CMD} run ${APP_PATH} \
    --server.address 0.0.0.0 \
    --server.port ${CONTAINER_PORT} \
    > ${LOG_DIR}/streamlit_\$(date +%Y%m%d_%H%M%S).log 2>&1
"

sleep 4

echo "[4/5] In-container health check (127.0.0.1:${CONTAINER_PORT})"
docker exec "${CONTAINER}" bash -lc "python3 - <<'PY'
import sys, urllib.request
url = 'http://127.0.0.1:${CONTAINER_PORT}'
try:
    with urllib.request.urlopen(url, timeout=5) as r:
        print('OK', r.status)
except Exception as e:
    print('FAIL', e)
    sys.exit(1)
PY"

echo "[5/5] Host-side health check (127.0.0.1:${HOST_PORT})"
python3 - <<PY
import sys, urllib.request
url = 'http://127.0.0.1:${HOST_PORT}'
try:
    with urllib.request.urlopen(url, timeout=5) as r:
        print('OK', r.status)
except Exception as e:
    print('FAIL', e)
    sys.exit(1)
PY

echo "Streamlit started successfully."
echo "Open: http://<server-ip>:${HOST_PORT}"
