#!/bin/bash
# 2026-10-27 The modules level after the module commits (MODULE_SPECS, forward wrappers, torch.Event).
# 1) end-to-end overhead as in ../exp_accept_p2.sh, at level 3: four opposite-phase in-run crossover pairs, Qwen3-0.6B
#    SFT DP8, GBS 32, overlap, metrics service, trace dir, REPORT_INTERVAL=10, 600 rollouts per run, the profiler
#    switched on and off every 2 rollouts. Code: a829f13 + modules_code.patch + ../patches/0001-0012.
#    Summary: python ../summarize_p2.py <log dir> modules_steps.csv m3-p 4 f19-c 2 (A/A runs as in ../exp_accept_f19.sh).
# 2) smoke on the PR branch with the module commits (8444de8 + ../patches/0001-0012): MoE-tiny DP8 x EP8, and
#    Qwen3-0.6B TP1 x PP4 with full activation recompute, 20 rollouts each, level 3, then check_module_trace.py.
# Usage: PART=crossover|smoke setsid nohup bash exp_modules.sh > logs/exp_modules-driver.log 2>&1 < /dev/null &
set -uo pipefail
umask 000

DIR=$(cd "$(dirname "$0")" && pwd)
: "${WORKDIR:?set WORKDIR to the directory holding relax_work/ and envs/}"
TASK_DIR=${WORKDIR}/relax_work/task11
PY=${WORKDIR}/envs/relax_venv/bin/python
: "${RELAX_CODE:?set RELAX_CODE to the code tree of the part being run}"; export RELAX_CODE
: "${PART:?set PART to crossover or smoke}"
export RAY_DEDUP_LOGS=0 STRAGGLER_LEVEL=3 TIMELINE=1 GBS=32
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
OVL=(--overlap-grad-reduce --overlap-param-gather)
NEXT_PORT=7300

trap 'exit 130' INT TERM HUP

wait_free() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 1024' | wc -l)" -gt 0 ]; do
    echo "$(date +%T) waiting for 8 free GPUs"; sleep 30
  done
}

listening() { ss -ltn | awk '{print $4}' | grep -q -E ":$1\$"; }

run() {  # run <tag> [launch args]; MODEL / TP / PP / AB_PHASE from the caller
  local tag=$1 rc log=${TASK_DIR}/logs/exp_accept_$1.log
  shift
  wait_free
  while listening "${NEXT_PORT}" || listening $((NEXT_PORT + 1000)); do NEXT_PORT=$((NEXT_PORT + 2)); done
  echo "$(date +%T) start ${tag}: ${MODEL} tp${TP} pp${PP} phase=${AB_PHASE:-0} gcs=${NEXT_PORT} args='$*'"
  RUN_TAG=${tag} STRAGGLER=1 GCS_PORT=${NEXT_PORT} DASH_PORT=$((NEXT_PORT + 1000)) \
    RAY_TMP=/tmp/ray_xtz_m3_${NEXT_PORT} bash "${TASK_DIR}/launch_sft.sh" "$@" > "${log}" 2>&1
  rc=$?
  echo "$(date +%T) ${tag} exit=${rc} $(grep -o 'training completed step [0-9]*' "${log}" | tail -1)"
  rm -rf "${TASK_DIR}/runs/sft-${MODEL}-tp${TP}pp${PP}-strag1-${tag}/ckpt"
}

check() {  # check <tag>; MODEL / TP / PP from the caller
  local traces=${TASK_DIR}/runs/sft-${MODEL}-tp${TP}pp${PP}-strag1-$1/timeline/straggler
  "${PY}" "${DIR}/check_module_trace.py" "${traces}"/straggler_trace_rollout*.json > "${TASK_DIR}/logs/check_$1.txt" 2>&1
  echo "$(date +%T) $1 trace check exit=$?"
}

echo "$(date +%T) part ${PART}, code ${RELAX_CODE}"
if [ "${PART}" = smoke ]; then
  export REPORT_INTERVAL=5 METRICS_SERVICE=0 NUM_ROLLOUT=20 AB_PERIOD=0
  MODEL=Qwen3-MoE-tiny-random TP=1 PP=1 run m363s-moe "${OVL[@]}" --expert-model-parallel-size 8
  MODEL=Qwen3-MoE-tiny-random TP=1 PP=1 check m363s-moe
  MODEL=Qwen3-0.6B TP=1 PP=4 run m363s-recompute "${OVL[@]}" \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  MODEL=Qwen3-0.6B TP=1 PP=4 check m363s-recompute
else
  export MODEL=Qwen3-0.6B TP=1 PP=1 METRICS_SERVICE=1 REPORT_INTERVAL=10
  export NUM_ROLLOUT=600 AB_PERIOD=2 GCSTAT=1 STRAGGLER_INTERVAL=1 COSTPROBE=0
  for i in 1 2 3 4; do
    AB_PHASE=0 run "m3-p${i}-ph0" "${OVL[@]}"
    AB_PHASE=1 run "m3-p${i}-ph1" "${OVL[@]}"
  done
fi
echo "$(date +%T) done"
