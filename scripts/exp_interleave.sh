#!/bin/bash
# In-run on/off crossover A/B for the straggler profiler overhead (8 GPUs, Qwen3-0.6B SFT DP8),
# followed by an A/A control pair (profiler off in both runs).
# Code: branch exp/task11-interleave (= feat/task11-straggler-profiler + experiment-only toggle); the
# Relax working tree must already be on it. Each pair runs phase 0 (blocks 0,2,4,... on) and phase 1
# (blocks 1,3,5,... on) with the same seed, so every 10-step block is measured once on and once off
# on identical data and run-level offsets cancel across even / odd blocks (summarize_interleave.py).
#
# The host runs a GPU queue daemon that dispatches jobs to cards with < 1 GiB used. A run's first
# minutes (Ray + model init) leave the cards nearly empty, so gpu_hold.py (memory only, no compute)
# holds every card between runs and until the run's first training step completes; it is released
# then because the training peak is close to 80 GB. A monitor kills gpu_occupy.py placeholders
# (never the daemon) and logs per-GPU memory to logs/expil_gpu_monitor.log every 10 s.
# Usage (tmux): PAIRS=3 NUM_ROLLOUT=300 CONTROL=1 bash exp_interleave.sh
set -uo pipefail
umask 000

TASK_DIR=/ytech_m2v4_hdd/xiatianze/relax_work/task11
RELAX_ROOT=/ytech_m2v4_hdd/xiatianze/Relax
PAIRS=${PAIRS:-3}
CONTROL=${CONTROL:-1}
export NUM_ROLLOUT=${NUM_ROLLOUT:-300}
MON=${TASK_DIR}/logs/expil_gpu_monitor.log
PY=/ytech_m2v4_hdd/xiatianze/envs/relax_venv/bin/python
mkdir -p "${TASK_DIR}/logs"
HOLD_PIDS=()

kill_occupiers() {
  for p in $(pgrep -f "[g]pu_occupy.py"); do
    cmd=$(tr '\0' ' ' < /proc/"$p"/cmdline 2>/dev/null)
    case "$cmd" in *gpu_queue.py*|*tmux*|"") ;; *) kill "$p" 2>/dev/null && echo "$(date +%T) killed occupier $p: $cmd" >> "$MON" ;; esac
  done
}

gpu_mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '; }

start_holders() {
  HOLD_PIDS=()
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g "$PY" "${TASK_DIR}/gpu_hold.py" > /dev/null 2>&1 &
    HOLD_PIDS+=($!)
  done
  sleep 20
  echo "$(date +%T) holders up: $(gpu_mem)"
}

stop_holders() {
  [ ${#HOLD_PIDS[@]} -gt 0 ] && kill "${HOLD_PIDS[@]}" 2>/dev/null
  wait "${HOLD_PIDS[@]}" 2>/dev/null
  HOLD_PIDS=()
  echo "$(date +%T) holders released"
}

(
  while true; do
    kill_occupiers
    echo "$(date +%T) $(gpu_mem)" >> "$MON"
    sleep 10
  done
) &
MON_PID=$!
trap 'kill ${MON_PID} 2>/dev/null; stop_holders' EXIT

# Before starting: every card must be free of foreign jobs (< 1 GiB), then hold them.
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 1024' | wc -l)" -gt 0 ]; do
  kill_occupiers; echo "$(date +%T) waiting for free GPUs: $(gpu_mem)"; sleep 30
done
start_holders

run() {  # run <tag> <straggler 0|1> <phase>
  # With holders up each card shows ~1.2 GiB; anything above 3 GiB is a foreign job.
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 3072' | wc -l)" -gt 0 ]; do
    echo "$(date +%T) foreign GPU use, waiting: $(gpu_mem)"; sleep 30
  done
  local log="${TASK_DIR}/logs/expil_$1.log"
  echo "$(date +%T) start $1"
  RUN_TAG=$1 STRAGGLER=$2 AB_PERIOD=10 AB_PHASE=$3 REPORT_INTERVAL=10 \
    bash "${TASK_DIR}/launch_sft.sh" > "$log" 2>&1 &
  local lpid=$!
  until grep -q "training completed step [1-9]" "$log" 2>/dev/null || ! kill -0 "$lpid" 2>/dev/null; do sleep 5; done
  echo "$(date +%T) $1 first step done: $(gpu_mem)"
  stop_holders
  wait "$lpid"
  echo "$(date +%T) $1 exit=$?"
  start_holders
}

GIT="git -c safe.directory=* -C ${RELAX_ROOT}"
echo "$(date +%T) host $(hostname -s) branch $($GIT branch --show-current) $($GIT rev-parse --short HEAD)"
for i in $(seq 1 "${PAIRS}"); do
  run il-p${i}-ph0 1 0
  run il-p${i}-ph1 1 1
done
if [ "${CONTROL}" = "1" ]; then
  run il-c1-ph0 0 0
  run il-c1-ph1 0 1
fi
stop_holders

source /ytech_m2v4_hdd/xiatianze/envs/relax_env.sh >/dev/null
python "${TASK_DIR}/summarize_interleave.py" il-p > "${TASK_DIR}/logs/expil_summary.md" 2>&1
[ "${CONTROL}" = "1" ] && python "${TASK_DIR}/summarize_interleave.py" il-c > "${TASK_DIR}/logs/expil_control_summary.md" 2>&1
echo "$(date +%T) done (logs/expil_summary.md, logs/expil_control_summary.md; switch Relax back to feat/task11-straggler-profiler by hand)"
