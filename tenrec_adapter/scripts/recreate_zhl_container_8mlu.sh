#!/bin/bash
set -euo pipefail

# Recreate docker container with 8 MLU devices and required port mappings.
#
# Usage:
#   bash tenrec_adapter/scripts/recreate_zhl_container_8mlu.sh
#   IMAGE=zhl_training_image:v1 NAME=zhl bash tenrec_adapter/scripts/recreate_zhl_container_8mlu.sh

NAME="${NAME:-zhl}"
IMAGE="${IMAGE:-zhl_training_image:v1}"
PROJECT_DIR="${PROJECT_DIR:-/root/zhl/x-algorithm}"
WORKDIR="${WORKDIR:-/root/zhl/x-algorithm}"
MLU_DEVICES="${MLU_DEVICES:-0,1,2,3,4,5,6,7}"
HOST_STREAMLIT_PORT="${HOST_STREAMLIT_PORT:-8501}"
CONTAINER_STREAMLIT_PORT="${CONTAINER_STREAMLIT_PORT:-8501}"
HOST_TB_PORT="${HOST_TB_PORT:-6006}"
CONTAINER_TB_PORT="${CONTAINER_TB_PORT:-6006}"

echo "[1/3] Stop and remove old container (if exists): ${NAME}"
docker stop "${NAME}" >/dev/null 2>&1 || true
docker rm "${NAME}" >/dev/null 2>&1 || true

echo "[2/3] Create new container with ports 8501/6006"
docker run -d \
  --name "${NAME}" \
  --restart unless-stopped \
  --ipc host \
  --pid host \
  --privileged \
  --shm-size=64g \
  -p "${HOST_STREAMLIT_PORT}:${CONTAINER_STREAMLIT_PORT}" \
  -p "${HOST_TB_PORT}:${CONTAINER_TB_PORT}" \
  -v /etc/cambricon:/etc/cambricon:ro \
  -v /usr/bin/cnmon:/usr/bin/cnmon:ro \
  -v "${PROJECT_DIR}:${WORKDIR}" \
  -w "${WORKDIR}" \
  -e MLU_VISIBLE_DEVICES="${MLU_DEVICES}" \
  -e LANG=C.UTF-8 \
  -e TZ=Asia/Shanghai \
  "${IMAGE}" \
  tail -f /dev/null

echo "[3/3] Verify container and ports"
docker ps --filter "name=${NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo "Container recreated."
