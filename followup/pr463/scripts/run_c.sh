#!/bin/bash
# 2026-10-27 End-to-end A/B for #463 (metrics client / service fix) on 8x A100: Qwen3-0.6B SFT DP8 (TP1 PP1), GBS 32,
# overlap, metrics service on (the upstream default), 120 rollouts per run.
#   before = a829f13 + #462 + experiment-only patches 0008-0011 (off unless set; 0011 gives per-rollout intervals)
#   after  = the same + #463's six commits (client.py, service.py and its tests are the only differences)
# Trees: code_pre = `git archive a829f13` + `git apply` of followup/perf/patches/0001, 0008-0011 of the evidence
# repository; code_post = the same + 0002-0007.
# Conditions: "proxy" keeps the container's http(s)_proxy variables, which every process here inherits from PID 1;
# "noproxy" removes them, as the other experiment drivers do. Two repetitions, the second in reverse order.
# Measured: the ms_probe shim times every HTTP request in every process (identical for both trees), [abstep] gives
# each rank's rollout interval, the coarse straggler profiler (window 10) gives who reaches the grad-sync last and how
# long the peers wait. Runs whose log reaches the last rollout are skipped; touch ./stop to end before the next run.
# Usage: setsid nohup bash run_c.sh >> logs/run_c-driver.log 2>&1 < /dev/null &
set -uo pipefail
umask 022

DIR=$(cd "$(dirname "$0")" && pwd)
# TASK_DIR holds launch_sft.sh / run_sft.sh (followup/perf/launch/ of the evidence repository) and runs/.
TASK_DIR=${TASK_DIR:?}
PY=${PY:-python3}
export RAY_DEDUP_LOGS=0
OVL=(--overlap-grad-reduce --overlap-param-gather)
NEXT_PORT=7400

trap 'exit 130' INT TERM HUP

wait_free() {
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 1024' | wc -l)" -gt 0 ]; do
    echo "$(date +%T) waiting for 8 free GPUs"; sleep 30
  done
}

listening() { ss -ltn | awk '{print $4}' | grep -q -E ":$1\$"; }

below_scan_limit() {  # on Ceph mounts, delete recursively only below 1,000,000 files
  [ "$(getfattr --only-values -n ceph.dir.rfiles "$1" 2> /dev/null || echo 0)" -lt 1000000 ]
}

run() {  # run <tag> <before|after> <proxy|noproxy>
  local tag=$1 arm=$2 cond=$3 rc log=${DIR}/logs/c_$1.log probe=${DIR}/probe/$1 ckpt
  local envcmd=(env) code=${DIR}/code_pre
  if [ "${arm}" = after ]; then code=${DIR}/code_post; fi
  if grep -q "training completed step $((NUM_ROLLOUT - 1))\b" "${log}" 2> /dev/null; then
    echo "$(date +%T) skip ${tag}: already done"; return
  fi
  if [ -e "${DIR}/stop" ]; then echo "$(date +%T) stop file, not starting ${tag}"; exit 0; fi
  if [ "${cond}" = noproxy ]; then envcmd=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY); fi
  wait_free
  while listening "${NEXT_PORT}" || listening $((NEXT_PORT + 1000)); do NEXT_PORT=$((NEXT_PORT + 2)); done
  rm -rf "${probe}"; mkdir -p "${probe}"
  echo "$(date +%T) start ${tag}: code=${arm} ${cond} gcs=${NEXT_PORT}"
  "${envcmd[@]}" RELAX_CODE="${code}" RELAX_MS_PROBE="${probe}" RELAX_MS_PROBE_SRC="${DIR}/probe_shim" \
    RUN_TAG=${tag} STRAGGLER=1 GCS_PORT=${NEXT_PORT} DASH_PORT=$((NEXT_PORT + 1000)) \
    RAY_TMP=/tmp/ray_ms463_c_${NEXT_PORT} bash "${TASK_DIR}/launch_sft.sh" "${OVL[@]}" > "${log}" 2>&1
  rc=$?
  echo "$(date +%T) ${tag} exit=${rc} $(grep -o 'training completed step [0-9]*' "${log}" | tail -1)"
  ckpt=${TASK_DIR}/runs/sft-${MODEL}-tp${TP}pp${PP}-strag1-${tag}/ckpt
  if [ -d "${ckpt}" ] && below_scan_limit "${ckpt}"; then rm -rf "${ckpt}"; fi
  rm -rf "/tmp/ray_ms463_c_${NEXT_PORT}"
}

export MODEL=Qwen3-0.6B TP=1 PP=1 GBS=32 METRICS_SERVICE=1 TIMELINE=0 NUM_ROLLOUT=120
export STRAGGLER_LEVEL=1 REPORT_INTERVAL=10 STRAGGLER_INTERVAL=1 AB_PERIOD=0 GCSTAT=0
echo "$(date +%T) before=${DIR}/code_pre after=${DIR}/code_post"

run c-r1-proxy-before before proxy
run c-r1-proxy-after after proxy
run c-r1-noproxy-before before noproxy
run c-r1-noproxy-after after noproxy
run c-r2-noproxy-after after noproxy
run c-r2-noproxy-before before noproxy
run c-r2-proxy-after after proxy
run c-r2-proxy-before before proxy

"${PY}" "${DIR}/summarize_c.py" "${DIR}" "${TASK_DIR}/runs" > "${DIR}/logs/c_summary.md" 2>&1
echo "$(date +%T) summary exit=$?"
echo "$(date +%T) all done"
