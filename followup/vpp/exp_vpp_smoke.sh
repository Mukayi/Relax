#!/bin/bash
# 2026-10-23 review-3 VPP smoke: the micro-batch / model-chunk identity under interleaved PP on Qwen3-0.6B
# TP1 x PP2 x DP4 with 2 virtual stages per PP rank (7 layers per chunk), overlap on, STRAGGLER_LEVEL=3,
# REPORT_INTERVAL=5, TIMELINE=1 (sampled steps 5, 10, 15 under the run's timeline/straggler/), metrics service off.
# Code: relax_exp_vpp = PR head + #462 (b57d72c), no experiment-only commits. TAG names the run (default vpp-modules); EXTRA_ARGS are appended to the launch arguments.
# Check: check_vpp_trace.py on the run's traces.
# Usage: setsid nohup bash exp_vpp_smoke.sh > logs/exp_vpp_smoke-driver.log 2>&1 < /dev/null &
set -uo pipefail
umask 000

TASK_DIR=${WORKDIR:?set WORKDIR to the directory with launch_sft.sh}
export RELAX_CODE=${RELAX_CODE:?set RELAX_CODE to the Relax checkout}
export RAY_DEDUP_LOGS=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
OVL=(--overlap-grad-reduce --overlap-param-gather)
VPP=(--num-virtual-stages-per-pipeline-rank 2)
PORT_BASE=$((6800 + RANDOM % 50 * 8))
RUN_NO=0
export MODEL=Qwen3-0.6B TP=1 PP=2 GBS=32 STRAGGLER=1 REPORT_INTERVAL=5 METRICS_SERVICE=0 TIMELINE=1

wait_free() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 1024' | wc -l)" -gt 0 ]; do
    echo "$(date +%T) waiting for 8 free GPUs"; sleep 30
  done
}

run() {  # run <tag> <extra launch args...>
  local tag=$1
  shift
  wait_free
  RUN_NO=$((RUN_NO + 1))
  local gcs=$((PORT_BASE + RUN_NO * 2)) dash=$((PORT_BASE + 1000 + RUN_NO * 2))
  local log=${TASK_DIR}/logs/exp_vpp_smoke_${tag}.log
  echo "$(date +%T) start ${tag}: NUM_ROLLOUT=${NUM_ROLLOUT} STRAGGLER_LEVEL=${STRAGGLER_LEVEL:--1} code $(git -c safe.directory='*' -C "${RELAX_CODE}" rev-parse --short HEAD)"
  RUN_TAG=${tag} GCS_PORT=${gcs} DASH_PORT=${dash} RAY_TMP=/tmp/ray_xtz_vpp_${gcs} \
    bash "${TASK_DIR}/launch_sft.sh" "$@" > "${log}" 2>&1
  local rc=$?
  rm -rf "${TASK_DIR}/runs/sft-${MODEL}-tp${TP}pp${PP}-strag${STRAGGLER}-${tag}/ckpt" "/tmp/ray_xtz_vpp_${gcs}"
  echo "$(date +%T) ${tag} exit=${rc} completed=$(grep -c "training completed step $((NUM_ROLLOUT - 1))" "${log}")"
}

NUM_ROLLOUT=20 STRAGGLER_LEVEL=3 run "${TAG:-vpp-modules}" "${OVL[@]}" "${VPP[@]}" ${EXTRA_ARGS:-}
echo "$(date +%T) all done"
