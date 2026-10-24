#!/bin/bash
# 2026-10-24 Overhead and long run on the final code for the bot's F19 (review of a829f13): count each rollout's
# whole cost in its own block.
# Code: ${RELAX_CODE} = a829f13 + #462 + #463 + the experiment-only sleep injection (inactive), in-run A/B toggle,
# GC counts and full-rollout interval (fb07997). Qwen3-0.6B SFT DP8, GBS 32, overlap on, metrics service on, trace
# dir on, profiler at its default level (level 2), REPORT_INTERVAL=10, same seed.
#   f19-p1 .. f19-p7  in-run crossover, 600 rollouts per run: ph0 has 10-rollout blocks 0, 2, 4, ... on, ph1 the
#                     others. STRAGGLER_INTERVAL=1 logs every rank's time from the end of one rollout to the end
#                     of the next (data wait, training, step_metrics, logging, flush, barrier) and switches the
#                     toggle only then; GCSTAT=1 logs every rank's collections.
#   f19-c1, f19-c2    the same with the profiler off in both runs (A/A)
#   f19-long          1200 rollouts with the profiler on throughout and no experiment logging; GPU memory and the
#                     summed RSS of the python processes sampled every 10 s
# Usage: setsid nohup bash exp_accept_f19.sh > logs/exp_accept_f19-driver.log 2>&1 < /dev/null &
set -uo pipefail
umask 000

: "${WORKDIR:?set WORKDIR to the directory holding relax_work/ and envs/}"
TASK_DIR=${WORKDIR}/relax_work/task11
: "${RELAX_CODE:?set RELAX_CODE to a829f13 with patches/ applied}"; export RELAX_CODE
export RAY_DEDUP_LOGS=0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
OVL=(--overlap-grad-reduce --overlap-param-gather)
MON=${TASK_DIR}/logs/exp_accept_f19-long_monitor.log
PORT_BASE=$((6800 + RANDOM % 50 * 8))
RUN_NO=0
MON_PID=
export MODEL=Qwen3-0.6B TP=1 PP=1 GBS=32
export METRICS_SERVICE=1 TIMELINE=1 REPORT_INTERVAL=10

gpu_mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '; }
py_rss() { ps -eo rss=,args= 2>/dev/null | awk '/[p]ython/ {s += $1} END {print int(s / 1024)}'; }
venv() { (set +u; source "${WORKDIR}/envs/relax_env.sh" >/dev/null 2>&1; "$@"); }
trap '[ -n "${MON_PID}" ] && kill "${MON_PID}" 2>/dev/null' EXIT
trap 'exit 130' INT TERM HUP

wait_free() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 1024' | wc -l)" -gt 0 ]; do
    echo "$(date +%T) waiting for 8 free GPUs"; sleep 30
  done
}

run() {  # run <tag> <straggler 0|1> [phase]
  local tag=$1 rc
  wait_free
  RUN_NO=$((RUN_NO + 1))
  local gcs=$((PORT_BASE + RUN_NO * 2)) log=${TASK_DIR}/logs/exp_accept_${tag}.log
  echo "$(date +%T) start ${tag}: straggler=$2 phase=${3:--} period=${AB_PERIOD} rollouts=${NUM_ROLLOUT}" \
    "gcstat=${GCSTAT} interval=${STRAGGLER_INTERVAL}"
  RUN_TAG=${tag} STRAGGLER=$2 AB_PHASE=${3:-0} GCS_PORT=${gcs} DASH_PORT=$((gcs + 1000)) \
    RAY_TMP=/tmp/ray_xtz_f19_${gcs} bash "${TASK_DIR}/launch_sft.sh" "${OVL[@]}" > "${log}" 2>&1
  rc=$?
  echo "$(date +%T) ${tag} exit=${rc} $(grep -o 'training completed step [0-9]*' "${log}" | tail -1)"
  rm -rf "${TASK_DIR}/runs/sft-${MODEL}-tp${TP}pp${PP}-strag$2-${tag}/ckpt"
}

pair() {  # pair <tag> <straggler 0|1>
  run "$1-ph0" "$2" 0
  run "$1-ph1" "$2" 1
}

echo "$(date +%T) code ${RELAX_CODE} $(git -c safe.directory='*' -C "${RELAX_CODE}" rev-parse --short HEAD)"
export NUM_ROLLOUT=600 AB_PERIOD=10 GCSTAT=1 STRAGGLER_INTERVAL=1
for i in 1 2 3; do pair "f19-p${i}" 1; done
pair f19-c1 0
for i in 4 5 6 7; do pair "f19-p${i}" 1; done
pair f19-c2 0

export NUM_ROLLOUT=1200 AB_PERIOD=0 GCSTAT=0 STRAGGLER_INTERVAL=0
( while true; do echo "$(date +%T) $(gpu_mem)rss=$(py_rss) MiB" >> "${MON}"; sleep 10; done ) &
MON_PID=$!
run f19-long 1
kill "${MON_PID}" 2>/dev/null
MON_PID=
echo "$(date +%T) done"
