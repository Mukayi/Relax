#!/bin/bash
# 2026-10-23 Follow-up to the overlap check of exp_accept_final.sh, whose profiler-on run (trace dir on, level 2)
# profiled rollouts 20-21 and so included the sampled step 20 with its deliberate device synchronize:
#   ac-ovl-on-l1     profiler on without a trace dir (level 1, no synchronize), torch.profiler on rollouts 20-21
#   ac-ovl-on-l2ns   profiler on with a trace dir (level 2), torch.profiler on rollouts 22-23 (no sampled step)
# Both against the profiler-off pair of exp_accept_final.sh (ac-ovl-off / ac-ovl-off2).
# Starts only after exp_accept_final.sh has exited, so the two queues never share the GPUs.
# Usage: setsid nohup bash exp_accept_ovl2.sh > logs/exp_accept_ovl2-driver.log 2>&1 < /dev/null &
set -uo pipefail
umask 000

TASK_DIR=${WORKDIR:?set WORKDIR}
export RELAX_CODE=${RELAX_CODE:?set RELAX_CODE}
export RAY_DEDUP_LOGS=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
OVL=(--overlap-grad-reduce --overlap-param-gather)
PY=python
PORT_BASE=$((7400 + RANDOM % 50 * 8))
RUN_NO=0
export MODEL=Qwen3-0.6B TP=1 PP=1 GBS=32 METRICS_SERVICE=1 NUM_ROLLOUT=40 REPORT_INTERVAL=10 STRAGGLER=1

while pgrep -f exp_accept_final.sh > /dev/null; do sleep 30; done
wait_free() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 1024' | wc -l)" -gt 0 ]; do
    echo "$(date +%T) waiting for 8 free GPUs"; sleep 30
  done
}

run() {  # run <tag> <first profiled rollout> [launch args]
  local tag=$1 first=$2; shift 2
  wait_free
  RUN_NO=$((RUN_NO + 1))
  local gcs=$((PORT_BASE + RUN_NO * 2)) log=${TASK_DIR}/logs/exp_accept_${tag}.log
  echo "$(date +%T) start ${tag}: timeline=${TIMELINE} profiled rollouts ${first}-$((first + 1))"
  RUN_TAG=${tag} GCS_PORT=${gcs} DASH_PORT=$((gcs + 1000)) RAY_TMP=/tmp/ray_xtz_ac2_${gcs} \
    bash "${TASK_DIR}/launch_sft.sh" "${OVL[@]}" --use-pytorch-profiler --profile-target train_overall \
      --profile-step-start "${first}" --profile-step-end "$((first + 1))" "$@" > "${log}" 2>&1
  echo "$(date +%T) ${tag} exit=$? $(grep -o 'training completed step [0-9]*' "${log}" | tail -1)"
  rm -rf "${TASK_DIR}/runs/sft-${MODEL}-tp${TP}pp${PP}-strag${STRAGGLER}-${tag}/ckpt"
}

traces() { echo "${RELAX_CODE}/traces/sft-${MODEL}-tp${TP}pp${PP}-strag$2-$1/train_trace"; }

TIMELINE=0 run ac-ovl-on-l1 20
TIMELINE=1 run ac-ovl-on-l2ns 22
for on in ac-ovl-on-l1 ac-ovl-on-l2ns; do
  "${PY}" "${TASK_DIR}/overlap_check.py" --off "$(traces ac-ovl-off 0)" --on "$(traces "${on}" 1)" \
    --off2 "$(traces ac-ovl-off2 0)" --out "${TASK_DIR}/logs/exp_accept_overlap_${on}.json" \
    > "${TASK_DIR}/logs/exp_accept_overlap_${on}.txt" 2>&1
  rc=$?
  echo "$(date +%T) ${on}: overlap_check exit ${rc}"
done
echo "$(date +%T) all done"
