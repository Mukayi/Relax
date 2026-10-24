#!/bin/bash
# 2026-10-24 Where does the profiler's ~0.5% at full functionality go? (exp_accept_f19.sh by block position: ~2 ms
# inside every rollout's training, ~10 ms per window at the sampled step.) COSTPROBE=1 times, on every rank and
# rollout, the segment timers, drain, sampled-step anchor, window close, end_step and step_metrics, and the step's
# log_perf_data, flush and barrier; STRAGGLER_INTERVAL=1 logs the rollout's full interval.
# Code: ${RELAX_CODE} (fb07997 + the cost probes d300d8f). Settings as exp_accept_f19.sh, 200 rollouts.
#   cost-on (profiler on throughout), cost-off (not installed)
# Usage: setsid nohup bash exp_accept_cost.sh > logs/exp_accept_cost-driver.log 2>&1 < /dev/null &
set -uo pipefail
umask 000

: "${WORKDIR:?set WORKDIR to the directory holding relax_work/ and envs/}"
TASK_DIR=${WORKDIR}/relax_work/task11
: "${RELAX_CODE:?set RELAX_CODE to the PR commit with patches/ applied}"; export RELAX_CODE
export RAY_DEDUP_LOGS=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
OVL=(--overlap-grad-reduce --overlap-param-gather)
export MODEL=Qwen3-0.6B TP=1 PP=1 GBS=32
export METRICS_SERVICE=1 TIMELINE=1 REPORT_INTERVAL=10
export NUM_ROLLOUT=200 AB_PERIOD=0 GCSTAT=0 STRAGGLER_INTERVAL=1 COSTPROBE=1
NEXT_PORT=7300

trap 'exit 130' INT TERM HUP

wait_free() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 1024' | wc -l)" -gt 0 ]; do
    echo "$(date +%T) waiting for 8 free GPUs"; sleep 30
  done
}

listening() { ss -ltn | awk '{print $4}' | grep -q -E ":$1\$"; }

run() {  # run <tag> <straggler 0|1>
  local tag=$1 rc log=${TASK_DIR}/logs/exp_accept_$1.log
  wait_free
  while listening "${NEXT_PORT}" || listening $((NEXT_PORT + 1000)); do NEXT_PORT=$((NEXT_PORT + 2)); done
  echo "$(date +%T) start ${tag}: straggler=$2 gcs=${NEXT_PORT}"
  RUN_TAG=${tag} STRAGGLER=$2 GCS_PORT=${NEXT_PORT} DASH_PORT=$((NEXT_PORT + 1000)) RAY_TMP=/tmp/ray_xtz_cost_${NEXT_PORT} \
    bash "${TASK_DIR}/launch_sft.sh" "${OVL[@]}" > "${log}" 2>&1
  rc=$?
  echo "$(date +%T) ${tag} exit=${rc} $(grep -o 'training completed step [0-9]*' "${log}" | tail -1)"
  rm -rf "${TASK_DIR}/runs/sft-${MODEL}-tp${TP}pp${PP}-strag$2-${tag}/ckpt"
}

echo "$(date +%T) code ${RELAX_CODE} $(git -c safe.directory='*' -C "${RELAX_CODE}" rev-parse --short HEAD)"
run cost-on 1
run cost-off 0
echo "$(date +%T) done"
