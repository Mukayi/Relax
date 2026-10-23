#!/bin/bash
# 2026-10-23 End-to-end overhead of the final code with reporting and overlap on (bot review of a829f13: the
# overhead was last measured on versions before the acceptance version). In-run on/off crossover as in
# exp_interleave.sh, plus an A/A control pair.
# Code: ${RELAX_CODE} = exp/accept-final (a829f13 + #462 + #463 + the sleep injection, inactive here) + the
# in-run A/B toggle ported to a829f13. Qwen3-0.6B SFT DP8, GBS 32, overlap on, metrics service on, trace dir on,
# profiler at its default level, REPORT_INTERVAL=10, 300 rollouts per run, same seed.
#   Pair i:  pf-p{i}-ph0 (10-rollout blocks 0, 2, 4, ... on) and pf-p{i}-ph1 (blocks 1, 3, 5, ... on)
#   Control: pf-c1-ph0 / pf-c1-ph1 with the profiler off in both runs (the method's own offset)
#   summarize_perf.py: per block on/off - 1, mean and 95% CI over the blocks after the first two.
# Usage: setsid nohup bash exp_accept_perf.sh > logs/exp_accept_perf-driver.log 2>&1 < /dev/null &
set -uo pipefail
umask 000

: "${WORKDIR:?set WORKDIR to the directory holding relax_work/ and envs/}"
TASK_DIR=${WORKDIR}/relax_work/task11
: "${RELAX_CODE:?set RELAX_CODE to a checkout of exp/accept-perf}"; export RELAX_CODE
export RAY_DEDUP_LOGS=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
OVL=(--overlap-grad-reduce --overlap-param-gather)
PAIRS=${PAIRS:-3}
PORT_BASE=$((6800 + RANDOM % 50 * 8))
RUN_NO=0
export MODEL=Qwen3-0.6B TP=1 PP=1 GBS=32
export METRICS_SERVICE=1 TIMELINE=1 REPORT_INTERVAL=10 NUM_ROLLOUT=300 AB_PERIOD=10

venv() { (set +u; source "${WORKDIR}/envs/relax_env.sh" >/dev/null 2>&1; "$@"); }
trap 'exit 130' INT TERM HUP

wait_free() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 1024' | wc -l)" -gt 0 ]; do
    echo "$(date +%T) waiting for 8 free GPUs"; sleep 30
  done
}

run() {  # run <tag> <straggler 0|1> <phase>
  local tag=$1 rc
  wait_free
  RUN_NO=$((RUN_NO + 1))
  local gcs=$((PORT_BASE + RUN_NO * 2)) log=${TASK_DIR}/logs/exp_accept_${tag}.log
  echo "$(date +%T) start ${tag}: straggler=$2 phase=$3 period=${AB_PERIOD} interval=${REPORT_INTERVAL}" \
    "metrics=${METRICS_SERVICE} timeline=${TIMELINE} rollouts=${NUM_ROLLOUT}"
  RUN_TAG=${tag} STRAGGLER=$2 AB_PHASE=$3 GCS_PORT=${gcs} DASH_PORT=$((gcs + 1000)) RAY_TMP=/tmp/ray_xtz_pf_${gcs} \
    bash "${TASK_DIR}/launch_sft.sh" "${OVL[@]}" > "${log}" 2>&1
  rc=$?
  echo "$(date +%T) ${tag} exit=${rc} $(grep -o 'training completed step [0-9]*' "${log}" | tail -1)"
  rm -rf "${TASK_DIR}/runs/sft-${MODEL}-tp${TP}pp${PP}-strag$2-${tag}/ckpt"
}

echo "$(date +%T) code ${RELAX_CODE} $(git -c safe.directory='*' -C "${RELAX_CODE}" rev-parse --short HEAD)"
for i in $(seq 1 "${PAIRS}"); do
  run "pf-p${i}-ph0" 1 0
  run "pf-p${i}-ph1" 1 1
done
run pf-c1-ph0 0 0
run pf-c1-ph1 0 1
venv python "${TASK_DIR}/summarize_perf.py" pf-p > "${TASK_DIR}/logs/exp_accept_perf_summary.md" 2>&1
venv python "${TASK_DIR}/summarize_perf.py" pf-c > "${TASK_DIR}/logs/exp_accept_perf_control_summary.md" 2>&1
echo "$(date +%T) done: logs/exp_accept_perf_summary.md, logs/exp_accept_perf_control_summary.md"
