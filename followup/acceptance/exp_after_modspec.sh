#!/bin/bash
# 2026-10-23 After exp_modspec_smoke.sh:
#   1. print_layers.py (now giving the provider its process groups) on three models, from the
#      feat/straggler-module-specs checkout;
#   2. the profiler-off pair for the overlap check of non-sampled steps: ac-ovl-off-22 / ac-ovl-off2-22, set up like
#      exp_accept_final.sh's ac-ovl-off but profiling rollouts 22-23, then overlap_check against ac-ovl-on-l2ns
#      (exp_accept_ovl2.sh), which profiled the same rollouts.
# Usage: setsid nohup bash exp_after_modspec.sh > logs/exp_after_modspec-driver.log 2>&1 < /dev/null &
set -uo pipefail
umask 000

TASK_DIR=${WORKDIR:?set WORKDIR}
SPEC_CODE=${SPEC_CODE:?set SPEC_CODE}
MODELS=${MODELS:?set MODELS}
export RELAX_CODE=${RELAX_CODE:?set RELAX_CODE}
export RAY_DEDUP_LOGS=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
OVL=(--overlap-grad-reduce --overlap-param-gather)
PY=python
PORT_BASE=$((8200 + RANDOM % 50 * 8))
RUN_NO=0

venv() { (set +u; source ${RELAX_ENV:?set RELAX_ENV} >/dev/null 2>&1; "$@"); }
while pgrep -f 'exp_modspec_smoke.sh' > /dev/null; do sleep 30; done
wait_free() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 1024' | wc -l)" -gt 0 ]; do
    echo "$(date +%T) waiting for 8 free GPUs"; sleep 30
  done
}

wait_free
for model in Qwen3-0.6B Qwen3-8B Qwen3-MoE-tiny-random; do
  CUDA_VISIBLE_DEVICES=0 venv python "${SPEC_CODE}/skills/straggler-module-trace/scripts/print_layers.py" \
    "${MODELS}/${model}" --layers 2 > "${TASK_DIR}/logs/exp_modspec_layers_${model}.txt" 2>&1
  rc=$?
  echo "$(date +%T) print_layers ${model} exit=${rc}"
done

run() {  # run <tag> <first profiled rollout>
  local tag=$1 first=$2
  wait_free
  RUN_NO=$((RUN_NO + 1))
  local gcs=$((PORT_BASE + RUN_NO * 2)) log=${TASK_DIR}/logs/exp_accept_${tag}.log
  echo "$(date +%T) start ${tag}: profiled rollouts ${first}-$((first + 1))"
  RUN_TAG=${tag} GCS_PORT=${gcs} DASH_PORT=$((gcs + 1000)) RAY_TMP=/tmp/ray_xtz_ac3_${gcs} \
    bash "${TASK_DIR}/launch_sft.sh" "${OVL[@]}" --use-pytorch-profiler --profile-target train_overall \
      --profile-step-start "${first}" --profile-step-end "$((first + 1))" > "${log}" 2>&1
  local rc=$?
  echo "$(date +%T) ${tag} exit=${rc} $(grep -o 'training completed step [0-9]*' "${log}" | tail -1)"
  rm -rf "${TASK_DIR}/runs/sft-${MODEL}-tp${TP}pp${PP}-strag${STRAGGLER}-${tag}/ckpt"
}

traces() { echo "${RELAX_CODE}/traces/sft-${MODEL}-tp${TP}pp${PP}-strag$2-$1/train_trace"; }

export MODEL=Qwen3-0.6B TP=1 PP=1 GBS=32 METRICS_SERVICE=1 NUM_ROLLOUT=40 REPORT_INTERVAL=10 TIMELINE=1 STRAGGLER=0
run ac-ovl-off-22 22
run ac-ovl-off2-22 22
"${PY}" "${TASK_DIR}/overlap_check.py" --off "$(traces ac-ovl-off-22 0)" --on "$(traces ac-ovl-on-l2ns 1)" \
  --off2 "$(traces ac-ovl-off2-22 0)" --out "${TASK_DIR}/logs/exp_accept_overlap_ac-ovl-on-l2ns-22.json" \
  > "${TASK_DIR}/logs/exp_accept_overlap_ac-ovl-on-l2ns-22.txt" 2>&1
rc=$?
echo "$(date +%T) overlap_check (steps 22-23) exit=${rc}"
echo "$(date +%T) all done"
