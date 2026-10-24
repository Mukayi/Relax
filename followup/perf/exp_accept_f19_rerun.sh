#!/bin/bash
# 2026-10-24 exp_accept_f19.sh runs f19-p5-ph0, f19-p5-ph1 and f19-p6-ph1 stopped at launch: their GCS ports were
# held by other services on the host. Same code and settings; the GCS / dashboard ports are checked free first.
# Usage: setsid nohup bash exp_accept_f19_rerun.sh > logs/exp_accept_f19_rerun-driver.log 2>&1 < /dev/null &
set -uo pipefail
umask 000

: "${WORKDIR:?set WORKDIR to the directory holding relax_work/ and envs/}"
TASK_DIR=${WORKDIR}/relax_work/task11
: "${RELAX_CODE:?set RELAX_CODE to a829f13 with patches/ applied}"; export RELAX_CODE
export RAY_DEDUP_LOGS=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
OVL=(--overlap-grad-reduce --overlap-param-gather)
export MODEL=Qwen3-0.6B TP=1 PP=1 GBS=32
export METRICS_SERVICE=1 TIMELINE=1 REPORT_INTERVAL=10
export NUM_ROLLOUT=600 AB_PERIOD=10 GCSTAT=1 STRAGGLER_INTERVAL=1
NEXT_PORT=7200

trap 'exit 130' INT TERM HUP

wait_free() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 1024' | wc -l)" -gt 0 ]; do
    echo "$(date +%T) waiting for 8 free GPUs"; sleep 30
  done
}

listening() { ss -ltn | awk '{print $4}' | grep -q -E ":$1\$"; }

free_port() {  # the next GCS port whose dashboard port (+1000) is free as well
  while listening "${NEXT_PORT}" || listening $((NEXT_PORT + 1000)); do NEXT_PORT=$((NEXT_PORT + 2)); done
  echo "${NEXT_PORT}"
  NEXT_PORT=$((NEXT_PORT + 2))
}

run() {  # run <tag> <phase>
  local tag=$1 rc gcs log=${TASK_DIR}/logs/exp_accept_$1.log
  wait_free
  gcs=$(free_port)
  echo "$(date +%T) start ${tag}: phase=$2 gcs=${gcs}"
  RUN_TAG=${tag} STRAGGLER=1 AB_PHASE=$2 GCS_PORT=${gcs} DASH_PORT=$((gcs + 1000)) RAY_TMP=/tmp/ray_xtz_f19_${gcs} \
    bash "${TASK_DIR}/launch_sft.sh" "${OVL[@]}" > "${log}" 2>&1
  rc=$?
  echo "$(date +%T) ${tag} exit=${rc} $(grep -o 'training completed step [0-9]*' "${log}" | tail -1)"
  rm -rf "${TASK_DIR}/runs/sft-${MODEL}-tp${TP}pp${PP}-strag1-${tag}/ckpt"
}

echo "$(date +%T) code ${RELAX_CODE} $(git -c safe.directory='*' -C "${RELAX_CODE}" rev-parse --short HEAD)"
run f19-p5-ph0 0
run f19-p5-ph1 1
run f19-p6-ph1 1
echo "$(date +%T) done"
