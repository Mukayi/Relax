#!/bin/bash
# 2026-10-27 Experiment B for #463: the service side, before / after, two repetitions in alternating order, CPU only on
# cores 120-127 (set CORES to change). The backends write to the shared file system like a training run's; the files are removed afterwards.
# Usage: setsid nohup bash run_b.sh >> logs/run_b-driver.log 2>&1 < /dev/null &
set -uo pipefail
umask 022

DIR=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
OUT_ROOT=${OUT_ROOT:?a directory on the file system a training run writes its TensorBoard logs to}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

below_scan_limit() {  # on Ceph mounts, delete recursively only below 1,000,000 files
  [ "$(getfattr --only-values -n ceph.dir.rfiles "$1" 2> /dev/null || echo 0)" -lt 1000000 ]
}

block() {  # block <before|after> <rep>
  local arm=$1 rep=$2 tree=${DIR}/code_pre
  if [ "${arm}" = after ]; then tree=${DIR}/code_post; fi
  if [ -s "${DIR}/logs/b_${arm}_r${rep}.csv" ]; then echo "$(date +%T) skip ${arm} r${rep}"; return; fi
  mkdir -p "${OUT_ROOT}/${arm}-r${rep}"
  echo "$(date +%T) start ${arm} r${rep}"
  PYTHONDONTWRITEBYTECODE=1 taskset -c "${CORES:-120-127}" "${PY}" "${DIR}/exp_b_server.py" "${arm}" "${tree}" "${rep}" \
    "${OUT_ROOT}/${arm}-r${rep}" "${DIR}/logs/b_${arm}_r${rep}.csv" 2>&1 | grep -v -i "warn\|Field(\|^ *$"
  echo "$(date +%T) ${arm} r${rep} exit=${PIPESTATUS[0]}"
  rm -rf /tmp/ray_ms463_b
}

block before 1
block after 1
block after 2
block before 2

if below_scan_limit "${OUT_ROOT}"; then rm -rf "${OUT_ROOT}"; fi
echo "$(date +%T) all done"
