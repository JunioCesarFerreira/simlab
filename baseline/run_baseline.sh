#!/usr/bin/env bash
# ============================================================================
# Script-based baseline orchestrator (§5.6 of the SimLab paper).
#
# Launches up to C_MAX Cooja simulations in parallel from a directory of
# pre-built input bundles, collects COOJA.testlog from each container into
# the results directory.
#
# This script is INTENTIONALLY MINIMAL and reflects common practice prior to
# the proposed architecture:
#   - no persistent state machine
#   - no event-driven coordination
#   - no automatic resume after failure
#   - no provenance tracking beyond COOJA.testlog filenames
#   - no idempotency: re-running overwrites previous results
#
# Comparison criteria are summarised in Table 6 of the manuscript.
#
# Usage:
#   ./run_baseline.sh [INPUTS_DIR] [RESULTS_DIR] [C_MAX]
#
# Defaults:
#   INPUTS_DIR  = ./inputs        (one subdir per simulation, each must
#                                  contain simulation.csc and optional
#                                  positions.dat + firmware/*.c)
#   RESULTS_DIR = ./results
#   C_MAX       = 10              (must be <= number of cooja* services in
#                                  docker-compose.baseline.yaml)
# ============================================================================
set -euo pipefail

INPUTS_DIR=${1:-./inputs}
RESULTS_DIR=${2:-./results}
C_MAX=${3:-10}

COMPOSE_FILE="$(dirname "$0")/docker-compose.baseline.yaml"
REMOTE_DIR="/opt/contiki-ng/tools/cooja"
SSH_USER="root"
SSH_PASS="root"  # default credentials of the Cooja image
JVM_XMS="${COOJA_JVM_XMS:-4g}"
JVM_XMX="${COOJA_JVM_XMX:-4g}"
SIM_TIMEOUT="${SIM_TIMEOUT_SEC:-3600}"

mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if ! command -v sshpass >/dev/null 2>&1; then
  echo "[baseline] ERROR: sshpass is required. Install via apt/brew." >&2
  exit 1
fi

if [[ ! -d "$INPUTS_DIR" ]]; then
  echo "[baseline] ERROR: inputs directory '$INPUTS_DIR' does not exist." >&2
  echo "[baseline]   Each subdirectory must contain a simulation.csc file." >&2
  exit 1
fi

mapfile -t SIM_DIRS < <(find "$INPUTS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ ${#SIM_DIRS[@]} -eq 0 ]]; then
  echo "[baseline] ERROR: no simulation bundles in '$INPUTS_DIR'." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Container fleet
# ---------------------------------------------------------------------------
echo "[baseline] starting $C_MAX cooja containers from $COMPOSE_FILE"
docker compose -f "$COMPOSE_FILE" up -d $(for i in $(seq 1 "$C_MAX"); do echo -n "cooja$i "; done)

# Give SSH a moment to be reachable
sleep 5

# ---------------------------------------------------------------------------
# Worker function — runs ONE simulation on ONE container.
# ---------------------------------------------------------------------------
run_one() {
  local worker_id=$1
  local sim_dir=$2
  local sim_name
  sim_name=$(basename "$sim_dir")
  local port=$((12230 + worker_id))
  local result_log="$RESULTS_DIR/${sim_name}.log"
  local elapsed_file="$RESULTS_DIR/${sim_name}.elapsed"

  echo "[baseline][worker $worker_id] -> $sim_name (port $port)"
  local t0
  t0=$(date +%s)

  # Upload inputs — plain scp, one file at a time
  for f in "$sim_dir"/*; do
    sshpass -p "$SSH_PASS" scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -P "$port" "$f" "$SSH_USER@localhost:$REMOTE_DIR/" 2>/dev/null \
      || { echo "[baseline][worker $worker_id] FAILED to upload $f"; return 1; }
  done

  # Execute Cooja headless. No status persistence, no progress tracking.
  local cmd="cd $REMOTE_DIR && /opt/java/openjdk/bin/java --enable-preview -Xms$JVM_XMS -Xmx$JVM_XMX -jar build/libs/cooja.jar --no-gui simulation.csc"
  timeout "$SIM_TIMEOUT" sshpass -p "$SSH_PASS" ssh \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -p "$port" "$SSH_USER@localhost" "$cmd" \
    > "$RESULTS_DIR/${sim_name}.stdout" 2> "$RESULTS_DIR/${sim_name}.stderr" \
    || echo "[baseline][worker $worker_id] WARNING: simulation $sim_name exited non-zero"

  # Fetch raw log
  sshpass -p "$SSH_PASS" scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -P "$port" "$SSH_USER@localhost:$REMOTE_DIR/COOJA.testlog" "$result_log" \
    2>/dev/null \
    || { echo "[baseline][worker $worker_id] WARNING: COOJA.testlog missing for $sim_name"; }

  local t1
  t1=$(date +%s)
  echo "$((t1 - t0))" > "$elapsed_file"
  echo "[baseline][worker $worker_id] done $sim_name (${result_log})"
}

# ---------------------------------------------------------------------------
# Crude job queue: dispatch each pending simulation to the next free worker.
# This is intentionally naive — no retry, no monitoring, no progress UI.
# ---------------------------------------------------------------------------
T_START=$(date +%s)
WORKER_PIDS=()
NEXT_WORKER=1
for sim_dir in "${SIM_DIRS[@]}"; do
  # Wait for a free slot
  while [[ ${#WORKER_PIDS[@]} -ge $C_MAX ]]; do
    NEW_PIDS=()
    for pid in "${WORKER_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        NEW_PIDS+=("$pid")
      fi
    done
    WORKER_PIDS=("${NEW_PIDS[@]}")
    [[ ${#WORKER_PIDS[@]} -ge $C_MAX ]] && sleep 1
  done

  run_one "$NEXT_WORKER" "$sim_dir" &
  WORKER_PIDS+=("$!")

  NEXT_WORKER=$((NEXT_WORKER + 1))
  [[ $NEXT_WORKER -gt $C_MAX ]] && NEXT_WORKER=1
done

# Wait for all remaining workers
wait
T_END=$(date +%s)

echo "[baseline] ============================================================"
echo "[baseline] simulations completed in $((T_END - T_START))s wall-clock."
echo "[baseline] results in $RESULTS_DIR/"
echo "[baseline] container fleet still running; stop with:"
echo "[baseline]   docker compose -f $COMPOSE_FILE down"
echo "[baseline] ============================================================"
