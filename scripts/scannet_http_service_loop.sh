#!/bin/bash
# Foreground supervisor: restarts the ScanNet HTTP render service every T seconds.
# Logs go to $PWD/scannet_http_service_<TS>/{supervisor.log,scannet_http_service_N.log}

MAX_WORKERS=${1:-32}
GPU_IDS=${2:-""}
OMP_CAP=${3:-1}
PORT=${4:-8767}
T=${5:-10800}            # restart interval in seconds (default: 3h)
BACKEND=${6:-open3d}     # open3d (mesh) or gsplat (3DGS)

SCRIPT_DIR="$(dirname "$0")"
SERVICE_SCRIPT="${SCRIPT_DIR}/scannet_http_service.sh"

RUN_DIR="$PWD/scannet_http_service_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
SUPERVISOR_LOG="${RUN_DIR}/supervisor.log"

log() { echo "[$(date)] $*" | tee -a "$SUPERVISOR_LOG"; }

log "Supervisor started (interval=${T}s MAX_WORKERS=${MAX_WORKERS} OMP_CAP=${OMP_CAP} PORT=${PORT} GPU_IDS=${GPU_IDS:-unset} BACKEND=${BACKEND})"
log "Run directory: ${RUN_DIR}"

CURRENT_PGID=""

kill_group() {
  local sig="$1" pgid="$2"
  if [ -n "$pgid" ] && kill -0 "-$pgid" 2>/dev/null; then
    kill "$sig" "-$pgid" 2>/dev/null || true
  fi
}

cleanup_and_exit() {
  trap '' INT TERM
  if [ -n "$CURRENT_PGID" ]; then
    log "Cleaning up service PGID=${CURRENT_PGID}"
    kill_group -TERM "$CURRENT_PGID"
    sleep 2
    kill_group -KILL "$CURRENT_PGID"
  fi
  exit 0
}
trap cleanup_and_exit INT TERM

SERVICE_COUNT=0
while true; do
  SERVICE_COUNT=$((SERVICE_COUNT + 1))
  SERVICE_LOG="${RUN_DIR}/scannet_http_service_${SERVICE_COUNT}.log"
  export LOG="$SERVICE_LOG"

  setsid "$SERVICE_SCRIPT" "$MAX_WORKERS" "$OMP_CAP" "$PORT" "$GPU_IDS" "$BACKEND" \
      >> "$SERVICE_LOG" 2>&1 &
  CHILD_PID=$!
  PGID=$(ps -o pgid= -p "$CHILD_PID" 2>/dev/null | tr -d ' ')
  PGID=${PGID:-$CHILD_PID}
  CURRENT_PGID="$PGID"
  log "Service #${SERVICE_COUNT} started PID=${CHILD_PID} PGID=${PGID} log=${SERVICE_LOG}"

  SECS=0
  while [ "$SECS" -lt "$T" ]; do
    if ! kill -0 "$CHILD_PID" 2>/dev/null; then
      log "Service #${SERVICE_COUNT} exited early (runtime=${SECS}s)"
      break
    fi
    sleep 1
    SECS=$((SECS + 1))
  done

  if kill -0 "$CHILD_PID" 2>/dev/null; then
    log "Timeout (${T}s), restarting service #${SERVICE_COUNT}"
    kill_group -TERM "$PGID"
    sleep 10
    kill -0 "$CHILD_PID" 2>/dev/null && kill_group -KILL "$PGID"
  fi

  CURRENT_PGID=""
done
