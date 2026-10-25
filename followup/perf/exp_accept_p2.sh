#!/bin/bash
# 2026-10-24 Final overhead: four opposite-phase in-run crossover pairs, 600 rollouts per run, full-rollout
# intervals and GC counts; the profiler switches on and off every 2 rollouts.
# Code: ${RELAX_CODE} = 891d0cf + #462 + #463 + the experiment-only sleep, A/B, GC and interval patches, plus the cost probes
# (off here). Qwen3-0.6B SFT DP8, GBS 32, overlap on, metrics service on, trace dir on, default level, REPORT_INTERVAL=10.
# Usage: setsid nohup bash exp_accept_p2.sh > logs/exp_accept_p2-driver.log 2>&1 < /dev/null &
set -uo pipefail
umask 000

: "${WORKDIR:?set WORKDIR to the directory holding relax_work/ and envs/}"
TASK_DIR=${WORKDIR}/relax_work/task11
: "${RELAX_CODE:?set RELAX_CODE to 891d0cf with patches/ applied}"; export RELAX_CODE
export RAY_DEDUP_LOGS=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
OVL=(--overlap-grad-reduce --overlap-param-gather)
export MODEL=Qwen3-0.6B TP=1 PP=1 GBS=32
export METRICS_SERVICE=1 TIMELINE=1 REPORT_INTERVAL=10
export NUM_ROLLOUT=600 AB_PERIOD=2 GCSTAT=1 STRAGGLER_INTERVAL=1 COSTPROBE=0
NEXT_PORT=7300

trap 'exit 130' INT TERM HUP

wait_free() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 1024' | wc -l)" -gt 0 ]; do
    echo "$(date +%T) waiting for 8 free GPUs"; sleep 30
  done
}

listening() { ss -ltn | awk '{print $4}' | grep -q -E ":$1\$"; }

run() {  # run <tag> <phase>
  local tag=$1 rc log=${TASK_DIR}/logs/exp_accept_$1.log
  wait_free
  while listening "${NEXT_PORT}" || listening $((NEXT_PORT + 1000)); do NEXT_PORT=$((NEXT_PORT + 2)); done
  echo "$(date +%T) start ${tag}: phase=$2 gcs=${NEXT_PORT}"
  RUN_TAG=${tag} STRAGGLER=1 AB_PHASE=$2 GCS_PORT=${NEXT_PORT} DASH_PORT=$((NEXT_PORT + 1000)) \
    RAY_TMP=/tmp/ray_xtz_p2_${NEXT_PORT} bash "${TASK_DIR}/launch_sft.sh" "${OVL[@]}" > "${log}" 2>&1
  rc=$?
  echo "$(date +%T) ${tag} exit=${rc} $(grep -o 'training completed step [0-9]*' "${log}" | tail -1)"
  rm -rf "${TASK_DIR}/runs/sft-${MODEL}-tp${TP}pp${PP}-strag1-${tag}/ckpt"
}

echo "$(date +%T) code ${RELAX_CODE} $(git -c safe.directory='*' -C "${RELAX_CODE}" rev-parse --short HEAD)"
for i in 1 2 3 4; do
  run "p2-p${i}-ph0" 0
  run "p2-p${i}-ph1" 1
done
echo "$(date +%T) done"
