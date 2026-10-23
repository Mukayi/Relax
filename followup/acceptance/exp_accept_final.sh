#!/bin/bash
# 2026-10-23 Acceptance reruns on the final code (bot review of a829f13: the numbers, the overlap and detection with
# reporting on were last measured on older versions).
# Code: relax_exp_accept = PR head a829f13 + #462 + #463 (the metrics-service fix, as in the acceptance version
# d8d8c39) + the experiment-only sleep injection. Qwen3-0.6B DP8, overlap on in every run, same seed.
#   Numbers (metrics service off, 40 rollouts; every run saves its grad norms, nothing asserts):
#     ac-num-off1, ac-num-off2   profiler off                                              -> A/A
#     ac-num-on                  profiler at its most intrusive: modules level, trace, a window every rollout -> A/B
#   Overlap (metrics service on, trace dir on, 40 rollouts, torch.profiler on rollouts 20-21):
#     ac-ovl-off, ac-ovl-on, ac-ovl-off2                                                   -> overlap_check.py
#   Detection with reporting on (metrics service on, trace dir on, 60 rollouts, REPORT_INTERVAL=10):
#     ac-loc-clean     no injection                                                        -> 0 alerts
#     ac-loc-sm5       GPU 5 hogged (gpu_hog.py, duty 1.0)                                 -> rank 5 slow_device
#     ac-loc-sleep3    rank 3 sleeps 200 ms at the start of every step                     -> rank 3 late_arrival
# Usage: setsid nohup bash exp_accept_final.sh > logs/exp_accept_final-driver.log 2>&1 < /dev/null &
set -uo pipefail
umask 000

TASK_DIR=${WORKDIR:?set WORKDIR}
export RELAX_CODE=${RELAX_CODE:?set RELAX_CODE}
export RAY_DEDUP_LOGS=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
OVL=(--overlap-grad-reduce --overlap-param-gather)
PROF=(--use-pytorch-profiler --profile-target train_overall --profile-step-start 20 --profile-step-end 21)
GRAD_ROOT=${TASK_DIR}/logs/accept_grad
PY=python
PORT_BASE=$((6800 + RANDOM % 50 * 8))
RUN_NO=0
HOG_PID=
export MODEL=Qwen3-0.6B TP=1 PP=1 GBS=32

venv() { (set +u; source ${RELAX_ENV:?set RELAX_ENV} >/dev/null 2>&1; "$@"); }
stop_hog() {
  [ -n "${HOG_PID}" ] || return 0
  pkill -TERM -P "${HOG_PID}" 2>/dev/null; kill "${HOG_PID}" 2>/dev/null; wait "${HOG_PID}" 2>/dev/null; HOG_PID=
}
trap stop_hog EXIT
trap 'exit 130' INT TERM HUP

wait_free() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 1024' | wc -l)" -gt 0 ]; do
    echo "$(date +%T) waiting for 8 free GPUs"; sleep 30
  done
}

run() {  # run <tag> [launch args]; NOWAIT=1 skips wait_free
  local tag=$1; shift
  [ "${NOWAIT:-0}" = "1" ] || wait_free
  RUN_NO=$((RUN_NO + 1))
  local gcs=$((PORT_BASE + RUN_NO * 2)) log=${TASK_DIR}/logs/exp_accept_${tag}.log
  echo "$(date +%T) start ${tag}: straggler=${STRAGGLER} level=${STRAGGLER_LEVEL:--1} interval=${REPORT_INTERVAL}" \
    "metrics=${METRICS_SERVICE} timeline=${TIMELINE} rollouts=${NUM_ROLLOUT} sleep=${INJECT_SLEEP_RANK:--1}"
  RUN_TAG=${tag} GCS_PORT=${gcs} DASH_PORT=$((gcs + 1000)) RAY_TMP=/tmp/ray_xtz_ac_${gcs} \
    bash "${TASK_DIR}/launch_sft.sh" "$@" > "${log}" 2>&1
  echo "$(date +%T) ${tag} exit=$? $(grep -o 'training completed step [0-9]*' "${log}" | tail -1)"
  rm -rf "${TASK_DIR}/runs/sft-${MODEL}-tp${TP}pp${PP}-strag${STRAGGLER}-${tag}/ckpt"
}

hog_run() {  # hog_run <gpu> <tag> [launch args]
  local gpu=$1; shift
  wait_free
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${gpu} venv python "${TASK_DIR}/gpu_hog.py" 8192 1.0 \
    > "${TASK_DIR}/logs/exp_accept_hog_gpu${gpu}.log" 2>&1 &
  HOG_PID=$!
  sleep 20
  if ! kill -0 "${HOG_PID}" 2>/dev/null; then echo "$(date +%T) hog on GPU ${gpu} failed to start"; HOG_PID=; return 1; fi
  echo "$(date +%T) hog pid ${HOG_PID} on GPU ${gpu}"
  NOWAIT=1 run "$@"
  stop_hog
}

trace_dir() {  # trace_dir <tag> <straggler>: torch.profiler writes ./traces/<tb_experiment_name>/train_trace
  local exp="sft-${MODEL}-tp${TP}pp${PP}-strag$2-$1"
  if [ -d "${RELAX_CODE}/traces/${exp}/train_trace" ]; then echo "${RELAX_CODE}/traces/${exp}/train_trace"; return; fi
  find /tmp/ray_xtz_ac_* -maxdepth 8 -type d -path "*traces/${exp}/train_trace" 2>/dev/null | head -1
}

echo "$(date +%T) code ${RELAX_CODE} $(git -c safe.directory='*' -C "${RELAX_CODE}" rev-parse --short HEAD)"

mkdir -p "${GRAD_ROOT}/off1" "${GRAD_ROOT}/off2" "${GRAD_ROOT}/on"
export METRICS_SERVICE=0 NUM_ROLLOUT=40 TIMELINE=0
for leg in off1 off2; do
  STRAGGLER=0 REPORT_INTERVAL=10 run "ac-num-${leg}" "${OVL[@]}" \
    --ci-save-grad-norm "${GRAD_ROOT}/${leg}/grad-r{rollout_id}-s{step_id}.pt"
done
STRAGGLER=1 REPORT_INTERVAL=1 TIMELINE=1 STRAGGLER_LEVEL=3 run ac-num-on "${OVL[@]}" \
  --ci-save-grad-norm "${GRAD_ROOT}/on/grad-r{rollout_id}-s{step_id}.pt"

export METRICS_SERVICE=1 NUM_ROLLOUT=40 TIMELINE=1 REPORT_INTERVAL=10
STRAGGLER=0 run ac-ovl-off "${OVL[@]}" "${PROF[@]}"
STRAGGLER=1 run ac-ovl-on "${OVL[@]}" "${PROF[@]}"
STRAGGLER=0 run ac-ovl-off2 "${OVL[@]}" "${PROF[@]}"
OFF=$(trace_dir ac-ovl-off 0); ON=$(trace_dir ac-ovl-on 1); OFF2=$(trace_dir ac-ovl-off2 0)
echo "$(date +%T) traces: off=${OFF} on=${ON} off2=${OFF2}"
"${PY}" "${TASK_DIR}/overlap_check.py" --off "${OFF}" --on "${ON}" --off2 "${OFF2}" \
  --out "${TASK_DIR}/logs/exp_accept_overlap_verdict.json" > "${TASK_DIR}/logs/exp_accept_overlap_verdict.txt" 2>&1
echo "$(date +%T) overlap verdict exit=$?"

export METRICS_SERVICE=1 NUM_ROLLOUT=60 TIMELINE=1 STRAGGLER=1 REPORT_INTERVAL=10
run ac-loc-clean "${OVL[@]}"
hog_run 5 ac-loc-sm5 "${OVL[@]}"
INJECT_SLEEP_RANK=3 INJECT_SLEEP_AT=step_start INJECT_SLEEP_MS=200 run ac-loc-sleep3 "${OVL[@]}"
echo "$(date +%T) all done"
