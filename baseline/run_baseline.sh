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
#                                  positions.dat + the firmware files,
#                                  flat in the same directory)
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
SIM_TIMEOUT="${SIM_TIMEOUT_SEC:-10800}"  # same default as the master-node
# What a run leaves in the container, removed before the next one so that it
# can be neither collected as that run's result nor reused by it. The same
# list as the master-node's (master-node/lib/transport/ssh.py).
GENERATED="build/cooja COOJA.testlog COOJA.log"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

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
remote() {  # remote PORT COMMAND
  sshpass -p "$SSH_PASS" ssh "${SSH_OPTS[@]}" -p "$1" "$SSH_USER@localhost" "$2"
}

run_one() {
  local worker_id=$1
  local sim_dir=$2
  local previous_dir=$3  # what this container ran last, if anything
  local sim_name
  sim_name=$(basename "$sim_dir")
  local port=$((12230 + worker_id))
  local result_log="$RESULTS_DIR/${sim_name}.log"
  local elapsed_file="$RESULTS_DIR/${sim_name}.elapsed"
  local exit_file="$RESULTS_DIR/${sim_name}.exit"
  # A log or exit code from an earlier invocation is not this run's.
  rm -f -- "$result_log" "$exit_file"

  echo "[baseline][worker $worker_id] -> $sim_name (port $port)"
  local t0
  t0=$(date +%s)

  # Start from the files this run uploads: the previous run's output and
  # inputs go first. Recompiling the firmware is part of the run, as on SimLab.
  local stale="$GENERATED simulation.csc positions.dat"
  if [[ -n "$previous_dir" ]]; then
    for f in "$previous_dir"/*; do stale+=" $(basename "$f")"; done
  fi
  remote "$port" "cd $REMOTE_DIR && rm -rf -- $stale" 2>/dev/null \
    || { echo "[baseline][worker $worker_id] FAILED to clean the container"; echo 255 > "$exit_file"; return 1; }

  # Upload inputs — plain scp, one file at a time
  for f in "$sim_dir"/*; do
    sshpass -p "$SSH_PASS" scp "${SSH_OPTS[@]}" \
      -P "$port" "$f" "$SSH_USER@localhost:$REMOTE_DIR/" 2>/dev/null \
      || { echo "[baseline][worker $worker_id] FAILED to upload $f"; echo 255 > "$exit_file"; return 1; }
  done

  # Execute Cooja headless. No status persistence, no progress tracking.
  # The time limit is applied in the container: a local one only ended the
  # ssh client, and the JVM went on running into the next simulation.
  local cmd="cd $REMOTE_DIR && timeout -k 30 $SIM_TIMEOUT /opt/java/openjdk/bin/java --enable-preview -Xms$JVM_XMS -Xmx$JVM_XMX -jar build/libs/cooja.jar --no-gui simulation.csc"
  local rc=0
  timeout $((SIM_TIMEOUT + 120)) sshpass -p "$SSH_PASS" ssh "${SSH_OPTS[@]}" \
    -p "$port" "$SSH_USER@localhost" "$cmd" \
    > "$RESULTS_DIR/${sim_name}.stdout" 2> "$RESULTS_DIR/${sim_name}.stderr" \
    || rc=$?
  echo "$rc" > "$exit_file"
  if [[ $rc -ne 0 ]]; then
    echo "[baseline][worker $worker_id] WARNING: simulation $sim_name exited with $rc"
    # Lost connection or a JVM past its limit: the container is not free
    # while one is still there.
    remote "$port" "pkill -KILL -f '[c]ooja.jar'" 2>/dev/null || true
  fi

  # Fetch raw log
  sshpass -p "$SSH_PASS" scp "${SSH_OPTS[@]}" \
    -P "$port" "$SSH_USER@localhost:$REMOTE_DIR/COOJA.testlog" "$result_log" \
    2>/dev/null \
    || { echo "[baseline][worker $worker_id] WARNING: COOJA.testlog missing for $sim_name"; }

  local t1
  t1=$(date +%s)
  echo "$((t1 - t0))" > "$elapsed_file"
  echo "[baseline][worker $worker_id] done $sim_name (${result_log})"
}

# ---------------------------------------------------------------------------
# Crude job queue: one simulation per container, on the first container whose
# simulation has ended.
# This is intentionally naive — no retry, no monitoring, no progress UI.
# ---------------------------------------------------------------------------
declare -A RUN_OF=()   # worker -> pid of the simulation it holds
declare -A LAST_OF=()  # worker -> bundle it ran last
FREE_WORKER=
wait_for_free_worker() {
  local w pid
  while :; do
    for w in $(seq 1 "$C_MAX"); do
      pid=${RUN_OF[$w]:-}
      if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        FREE_WORKER=$w
        return
      fi
    done
    sleep 1
  done
}

T_START=$(date +%s)
for sim_dir in "${SIM_DIRS[@]}"; do
  wait_for_free_worker
  run_one "$FREE_WORKER" "$sim_dir" "${LAST_OF[$FREE_WORKER]:-}" &
  RUN_OF[$FREE_WORKER]=$!
  LAST_OF[$FREE_WORKER]=$sim_dir
done

# Wait for all remaining workers
wait
T_END=$(date +%s)

COMPLETED=0
for sim_dir in "${SIM_DIRS[@]}"; do
  sim_name=$(basename "$sim_dir")
  if [[ "$(cat "$RESULTS_DIR/${sim_name}.exit" 2>/dev/null)" == 0 && -s "$RESULTS_DIR/${sim_name}.log" ]]; then
    COMPLETED=$((COMPLETED + 1))
  fi
done

echo "[baseline] ============================================================"
echo "[baseline] simulations completed in $((T_END - T_START))s wall-clock."
echo "[baseline] $COMPLETED of ${#SIM_DIRS[@]} simulations exited 0 with a log."
echo "[baseline] results in $RESULTS_DIR/"
echo "[baseline] container fleet still running; stop with:"
echo "[baseline]   docker compose -f $COMPOSE_FILE down"
echo "[baseline] ============================================================"
