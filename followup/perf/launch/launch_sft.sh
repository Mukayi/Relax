#!/bin/bash
# Start a PRIVATE single-node Ray head (non-default ports, own temp dir), run
# run_sft.sh against it, then tear down only that head.
#
# Why not scripts/entrypoint/local.sh: it runs `pkill -9 python` / `ray stop --force`,
# which on a shared dev node kills other people's work. ray-job.sh also kills
# Relax workers cluster-wide. This launcher never sends a signal to anything it
# did not start: `ray start --block` spawns a reaper that tears down exactly the
# processes of this head when the blocking process exits.
#
# Usage:
#   bash launch_sft.sh                       # 60-step Qwen3-0.6B DP8 smoke
#   STRAGGLER=1 REPORT_INTERVAL=5 bash launch_sft.sh
#   MODEL=Qwen3-4B TP=2 PP=2 GBS=32 bash launch_sft.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 bash launch_sft.sh
#   PEER=<ip> PEER_PORT=<mpi sshd port> NUM_GPUS=16 bash launch_sft.sh   # + a Ray worker on the MPI peer
# Env knobs are documented in run_sft.sh; this file adds:
#   STRAGGLER=0|1, REPORT_INTERVAL, TIMELINE=0|1, GCS_PORT, DASH_PORT, RUN_TAG,
#   PEER / PEER_PORT (two nodes: the head binds this host's first IP, the worker is started over root ssh,
#   CURSOR.md 6.1), and NCCL_NET_PLUGIN / NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME / NCCL_IB_DISABLE /
#   NCCL_IB_HCA / NCCL_DEBUG, forwarded to the Ray workers when set

: "${WORKDIR:?set WORKDIR to the directory holding Relax/, relax_work/, relax_assets/, envs/ and external/}"
set -eo pipefail

source ${WORKDIR}/envs/relax_env.sh   # venv + LD_LIBRARY_PATH + cd $RELAX_ROOT
# RELAX_CODE: run another checkout (e.g. a worktree with experiment-only patches) instead of Relax/.
if [ -n "${RELAX_CODE:-}" ]; then export RELAX_ROOT=${RELAX_CODE}; cd "${RELAX_ROOT}"; fi
TASK_DIR=${WORKDIR}/relax_work/task11

export MODEL=${MODEL:-Qwen3-0.6B} TP=${TP:-1} PP=${PP:-1} NUM_GPUS=${NUM_GPUS:-8}
STRAGGLER=${STRAGGLER:-0}
REPORT_INTERVAL=${REPORT_INTERVAL:-10}
TIMELINE=${TIMELINE:-0}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
export EXP_NAME=${EXP_NAME:-sft-${MODEL}-tp${TP}pp${PP}-strag${STRAGGLER}-${RUN_TAG}}
export RUN_DIR=${TASK_DIR}/runs/${EXP_NAME}
mkdir -p "${RUN_DIR}"

GCS_PORT=${GCS_PORT:-6479}
DASH_PORT=${DASH_PORT:-8365}
# Ray's workers take a block of ephemeral ports (10002-19999 on this box), and a
# GCS/dashboard port inside that range makes the head die with
# "port number ... is used by other components" or a bare GCS timeout after two
# minutes. Fail here instead: the wait loop below costs 45 min per attempt.
for _p in "${GCS_PORT}" "${DASH_PORT}"; do
  if [ "${_p}" -ge 10000 ] && [ "${_p}" -le 19999 ]; then
    echo "GCS_PORT/DASH_PORT ${_p} falls inside Ray's worker port range 10002-19999; pick ports below 10000" >&2
    exit 4
  fi
done
RAY_TMP=${RAY_TMP:-/tmp/ray_xtz_task11_${GCS_PORT}}

# ── environment (mirrors scripts/entrypoint/local.sh without the cleanup) ────
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RELAX=${RELAX_ROOT}
export MEGATRON=${WORKDIR}/external/Megatron-LM
export PYTHONPATH=${RELAX}:${MEGATRON}:${PYTHONPATH:-}
# Experiment-only: instrument HTTPAdapter.send via a sitecustomize shim so the
# metrics-service A/B can be measured without patching either checkout. Both the
# driver process and the Ray workers need it, so it goes on PYTHONPATH (which is
# forwarded in RUNTIME_ENV_JSON below) and RELAX_MS_PROBE names the output dir.
if [ -n "${RELAX_MS_PROBE:-}" ]; then
  export PYTHONPATH=${RELAX_MS_PROBE_SRC}:${PYTHONPATH}
  echo "=== metrics probe enabled: ${RELAX_MS_PROBE_SRC} -> ${RELAX_MS_PROBE} ==="
fi
export MODEL_CONFIG_DIR=${RELAX_ROOT}/scripts/models
HEAD_IP=127.0.0.1
if [ -n "${PEER:-}" ]; then HEAD_IP=$(hostname -I | awk '{print $1}'); fi
export MASTER_ADDR=${HEAD_IP}
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "${NVLINK_COUNT}" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi

# ── private Ray head ────────────────────────────────────────────────────────
if ss -ltn 2>/dev/null | grep -qE ":(${GCS_PORT}|${DASH_PORT}) "; then
  echo "port ${GCS_PORT} or ${DASH_PORT} already in use; pick GCS_PORT/DASH_PORT" >&2
  exit 3
fi
echo "=== starting private ray head gcs=${GCS_PORT} dashboard=${DASH_PORT} tmp=${RAY_TMP} gpus=${NUM_GPUS} ==="
HEAD_GPUS=${NUM_GPUS}
if [ -n "${PEER:-}" ]; then HEAD_GPUS=$((NUM_GPUS / 2)); fi
setsid ray start --head --block \
    --node-ip-address "${HEAD_IP}" --port "${GCS_PORT}" \
    --dashboard-host 127.0.0.1 --dashboard-port "${DASH_PORT}" \
    --num-gpus "${HEAD_GPUS}" --disable-usage-stats \
    --temp-dir "${RAY_TMP}" > "${RUN_DIR}/ray_head.log" 2>&1 &
RAY_HEAD_PID=$!
PEER_SSH=(ssh -p "${PEER_PORT:-22}" -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
  -o ConnectTimeout=10 "root@${PEER:-}")
RAY_WORKER_PID=
cleanup() {
  if [ -n "${RAY_WORKER_PID}" ]; then
    echo "=== stopping ray worker on ${PEER} (pid ${RAY_WORKER_PID}) ==="
    "${PEER_SSH[@]}" "kill -TERM ${RAY_WORKER_PID}; for _ in \$(seq 1 30); do kill -0 ${RAY_WORKER_PID} 2> /dev/null || exit 0; sleep 1; done; kill -KILL ${RAY_WORKER_PID}" || true
  fi
  echo "=== stopping private ray head (pid ${RAY_HEAD_PID}) ==="
  kill -TERM "${RAY_HEAD_PID}" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "${RAY_HEAD_PID}" 2>/dev/null || break; sleep 1; done
  kill -KILL "${RAY_HEAD_PID}" 2>/dev/null || true
}
trap cleanup EXIT
export RAY_ADDRESS=${HEAD_IP}:${GCS_PORT}
export RAY_DASHBOARD_URL=http://127.0.0.1:${DASH_PORT}
for _ in $(seq 1 60); do
  if ray status --address "${RAY_ADDRESS}" >/dev/null 2>&1 && curl -sf "${RAY_DASHBOARD_URL}/api/version" >/dev/null 2>&1; then break; fi
  sleep 2
done
if [ -n "${PEER:-}" ]; then
  echo "=== starting ray worker on ${PEER}:${PEER_PORT:-22} -> ${RAY_ADDRESS} gpus=$((NUM_GPUS - HEAD_GPUS)) ==="
  RAY_WORKER_PID=$("${PEER_SSH[@]}" "source ${TASK_DIR}/tools/pid1_env.sh; \
source ${WORKDIR}/envs/relax_env.sh > /dev/null; umask 000; \
unset CUDA_VISIBLE_DEVICES; setsid nohup ray start --block --address ${RAY_ADDRESS} --node-ip-address ${PEER} \
--num-gpus $((NUM_GPUS - HEAD_GPUS)) --disable-usage-stats --temp-dir ${RAY_TMP} \
> ${RUN_DIR}/ray_worker.log 2>&1 < /dev/null & echo \$!")
  for _ in $(seq 1 60); do
    ray status --address "${RAY_ADDRESS}" 2>/dev/null | grep -qE "/${NUM_GPUS}(\.0)? GPU" && break
    sleep 2
  done
fi
ray status --address "${RAY_ADDRESS}"

# ── runtime env for the Ray job (workers do not inherit this shell) ─────────
# NCCL channel limits are only passed when set: an empty value is not a valid NCCL setting.
NCCL_CHANNELS_JSON=""
if [ -n "${NCCL_MAX_NCHANNELS:-}" ]; then
  NCCL_CHANNELS_JSON="${NCCL_CHANNELS_JSON}
   \"NCCL_MAX_NCHANNELS\": \"${NCCL_MAX_NCHANNELS}\","
fi
if [ -n "${NCCL_MIN_NCHANNELS:-}" ]; then
  NCCL_CHANNELS_JSON="${NCCL_CHANNELS_JSON}
   \"NCCL_MIN_NCHANNELS\": \"${NCCL_MIN_NCHANNELS}\","
fi
for _v in NCCL_NET_PLUGIN NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_DISABLE NCCL_IB_HCA NCCL_DEBUG; do
  if [ -n "${!_v:-}" ]; then NCCL_CHANNELS_JSON="${NCCL_CHANNELS_JSON}
   \"${_v}\": \"${!_v}\","; fi
done
TB_DIR=${RUN_DIR}/tensorboard
if [ "${TIMELINE}" = "1" ]; then export TIMELINE_DUMP_DIR=${RUN_DIR}/timeline; fi
export RUNTIME_ENV_JSON="{
\"worker_process_setup_hook\": \"relax.utils.logging_utils.install_asyncio_noise_filter\",
\"env_vars\": {
   \"PYTHONUNBUFFERED\": \"1\",
   \"PYTHONPATH\": \"${PYTHONPATH}\",
   \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH}\",
   \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
   \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
   \"OMP_NUM_THREADS\": \"${OMP_NUM_THREADS:-24}\",
   \"MKL_NUM_THREADS\": \"${MKL_NUM_THREADS:-24}\",
   \"OPENBLAS_NUM_THREADS\": \"${OPENBLAS_NUM_THREADS:-24}\",
   \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
   \"NCCL_P2P_DISABLE\": \"${NCCL_P2P_DISABLE:-0}\",${NCCL_CHANNELS_JSON}
   \"HF_HUB_OFFLINE\": \"1\",
   \"WANDB_MODE\": \"disabled\",
   \"TENSORBOARD_DIR\": \"${TB_DIR}\",
   \"RELAX_STRAGGLER_PROFILER\": \"${STRAGGLER}\",
   \"RELAX_STRAGGLER_REPORT_INTERVAL\": \"${REPORT_INTERVAL}\",
   \"RELAX_STRAGGLER_PERSIST_WINDOWS\": \"${PERSIST_WINDOWS:-3}\",
   \"RELAX_STRAGGLER_GATHER_TIMEOUT_S\": \"${STRAGGLER_GATHER_TIMEOUT_S:-60}\",
   \"RELAX_STRAGGLER_INJECT_MAX_PENDING\": \"${INJECT_MAX_PENDING:-0}\",
   \"RELAX_STRAGGLER_INJECT_GATHER_SKIP_RANK\": \"${INJECT_GATHER_SKIP_RANK:--1}\",
   \"RELAX_STRAGGLER_INJECT_REPORT_RAISE\": \"${INJECT_REPORT_RAISE:-0}\",
   \"STRAGGLER_AB_PERIOD\": \"${AB_PERIOD:-0}\",
   \"STRAGGLER_AB_PHASE\": \"${AB_PHASE:-0}\",
   \"STRAGGLER_AB_LEVELS\": \"${AB_LEVELS:-}\",
   \"RELAX_STRAGGLER_LEVEL\": \"${STRAGGLER_LEVEL:--1}\",
   \"STRAGGLER_AB_LEVELS\": \"${AB_LEVELS:-}\",
   \"RELAX_STRAGGLER_LEVEL\": \"${STRAGGLER_LEVEL:--1}\",
   \"STRAGGLER_INJECT_SLEEP_RANK\": \"${INJECT_SLEEP_RANK:--1}\",
   \"STRAGGLER_INJECT_SLEEP_MS\": \"${INJECT_SLEEP_MS:-0}\",
   \"STRAGGLER_INJECT_SLEEP_AT\": \"${INJECT_SLEEP_AT:-rollout_end}\",
   \"STRAGGLER_INJECT_SLEEP_START\": \"${INJECT_SLEEP_START:-0}\",
   \"STRAGGLER_INJECT_STOP\": \"${INJECT_STOP:-1073741824}\",
   \"STRAGGLER_INJECT_GC_RANK\": \"${INJECT_GC_RANK:--1}\",
   \"STRAGGLER_INJECT_GC_OBJECTS\": \"${INJECT_GC_OBJECTS:-2000000}\",
   \"STRAGGLER_INJECT_GC_EVERY\": \"${INJECT_GC_EVERY:-1}\",
   \"STRAGGLER_INJECT_TOKENS_RANK\": \"${INJECT_TOKENS_RANK:--1}\",
   \"STRAGGLER_INJECT_TOKENS_FACTOR\": \"${INJECT_TOKENS_FACTOR:-1}\",
   \"STRAGGLER_INJECT_GATHER_FAIL_RANK\": \"${INJECT_GATHER_FAIL_RANK:--1}\",
   \"STRAGGLER_INJECT_GATHER_FAIL_COUNT\": \"${INJECT_GATHER_FAIL_COUNT:-0}\",
   \"GCSTAT\": \"${GCSTAT:-0}\",
   \"STRAGGLER_INTERVAL\": \"${STRAGGLER_INTERVAL:-0}\",
   \"RELAX_MS_PROBE\": \"${RELAX_MS_PROBE:-}\"
}
}"
export RELAX_ENTRYPOINT_MODE="local"

echo "=== EXP ${EXP_NAME}: MODEL=${MODEL} TP=${TP} PP=${PP} GPUS=${NUM_GPUS} STRAGGLER=${STRAGGLER} interval=${REPORT_INTERVAL} timeline=${TIMELINE} ==="
echo "run dir: ${RUN_DIR}"
bash "${TASK_DIR}/run_sft.sh" "$@"
