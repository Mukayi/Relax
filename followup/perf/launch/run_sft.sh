#!/bin/bash
# Task-11 SFT run script (derived from Relax/scripts/training/sft/run-qwen3-0.6B-math-8xgpu.sh).
#
# Differences from upstream, all so the run is self-contained on a shared dev node:
#   * no ClearML (no server here); TensorBoard via TENSORBOARD_DIR from launch_sft.sh
#   * assets under relax_assets/ (HF weights + OpenMathReasoning-mini parquet)
#   * --num-rollout caps the run (default 60 steps) and --save-interval is huge (no checkpoints)
#   * MODEL / TP / PP / GBS / MAX_TOKENS_PER_GPU / EXP_NAME are env-overridable
#
# Must be launched through launch_sft.sh, which starts a private Ray head and sets
# RELAX_ENTRYPOINT_MODE so this script does NOT source scripts/entrypoint/local.sh
# (that file pkill -9's every python on the node).

set -ex
set -o pipefail

: "${RELAX_ENTRYPOINT_MODE:?run through launch_sft.sh}"
: "${RUNTIME_ENV_JSON:?run through launch_sft.sh}"
: "${RAY_DASHBOARD_URL:?run through launch_sft.sh}"

RELAX_ROOT=${RELAX_ROOT:-${WORKDIR}/Relax}
ASSETS=${ASSETS:-${WORKDIR}/relax_assets}
RUN_DIR=${RUN_DIR:?launch_sft.sh sets RUN_DIR}

MODEL=${MODEL:-Qwen3-0.6B}          # Qwen3-0.6B | Qwen3-4B (must exist under $ASSETS/models)
TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}
GBS=${GBS:-32}
# Upstream uses 20480; here a 20k-token micro-batch OOMs on the 80 GB A800 at step 4
# (10.5 GiB fp32 [tokens, vocab=151936] logits grad in backward on top of ~52 GiB live).
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
NUM_ROLLOUT=${NUM_ROLLOUT:-60}
NUM_GPUS=${NUM_GPUS:-8}
EXP_NAME=${EXP_NAME:-task11-sft-${MODEL}-tp${TP}pp${PP}}

case "${MODEL}" in
  Qwen3-0.6B) source "${RELAX_ROOT}/scripts/models/qwen3-0.6B.sh" ;;
  Qwen3-4B)   source "${RELAX_ROOT}/scripts/models/qwen3-4B.sh" ;;
  Qwen3-8B)   source "${RELAX_ROOT}/scripts/models/qwen3-8B.sh" ;;
  Qwen3-MoE-tiny-random) source "$(dirname "$0")/models/qwen3-moe-tiny.sh" ;;
  *) echo "unknown MODEL=${MODEL}" >&2; exit 2 ;;
esac

PROMPT_DATA="${ASSETS}/data/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet"

CKPT_ARGS=(
   --hf-checkpoint "${ASSETS}/models/${MODEL}"
   --ref-load "${ASSETS}/models/${MODEL}"
   --megatron-to-hf-mode bridge
   --save "${RUN_DIR}/ckpt"
   --save-interval 1000000
   --num-epoch 10
   --num-rollout "${NUM_ROLLOUT}"
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key problem
   --label-key generated_solution
   --global-batch-size "${GBS}"
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --sft-oversize-strategy skip
   --per-rank-fetch
   --sft-prefetch-num-workers 8
   --sft-prefetch-buffer-size 256
)
# The actor rejects --sft-async-prepack unless PP=1 and CP=1 (_validate_sft_prepack_pipeline).
if [ "${PP}" = "1" ] && [ "${CP}" = "1" ]; then SFT_ARGS+=(--sft-async-prepack); fi
# BALANCE_DATA=0 drops --balance-data (data_imbalance injection); upstream recipe keeps it.
if [ "${BALANCE_DATA:-1}" = "1" ]; then SFT_ARGS+=(--balance-data); fi

EVAL_ARGS=(
    --eval-size 0.01
    --eval-interval 1000000
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP}"
   --pipeline-model-parallel-size "${PP}"
   --context-parallel-size "${CP}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --no-rope-fusion
   --colocate
   # --colocate defaults offload_train=True, which LD_PRELOADs torch_memory_saver
   # (CI-image only, not in this venv). SFT here has no rollout engine to make room
   # for, so keep the train state resident like the non-offload upstream SFT runs.
   --no-offload-train
   --no-offload-rollout
   --cross-entropy-loss-fusion
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
)

LOG_ARGS=(
   --use-tensorboard
   --tb-project-name task11
   --tb-experiment-name "${EXP_NAME}"
)
# Relax routes logging through the Ray Serve metrics service by default (--use-metrics-service
# defaults to True). Its client does a blocking requests.post per log call on rank 0's
# training thread (p90 260 ms, max 2.6 s here) which made rank 0 reach every grad-sync
# 0.4-2.6 s late (see NOTES.md §3.6). Off by default so the A/B overhead runs measure
# the profiler, not that; METRICS_SERVICE=1 restores the upstream behaviour.
if [ "${METRICS_SERVICE:-0}" = "1" ]; then
   LOG_ARGS+=(--use-metrics-service)
else
   LOG_ARGS+=(--no-use-metrics-service)
fi
if [ -n "${TIMELINE_DUMP_DIR:-}" ]; then
   LOG_ARGS+=(--timeline-dump-dir "${TIMELINE_DUMP_DIR}")
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # The image's flash_attn_2_cuda.so only carries sm_100 kernels (B300 image); A800 is sm_80
   # -> fixed by the sm_80 rebuild installed in the venv (relax_work/fa_build). TE cuDNN fused attn
   # on sm_80 only supports bshd, not the thd packing SFT uses, so "fused" is not a fallback here.
   --attention-backend "${ATTN_BACKEND:-flash}"
   --use-health-check
   # The image's apex fused_weight_gradient_mlp_cuda.so is ABI-incompatible with torch 2.13
   # (undefined at::cuda::getCurrentCUDABlasHandle), same class of issue as TE; the NPU
   # scripts run without the fusion too. Identical for with/without-profiler runs.
   --no-gradient-accumulation-fusion
)

mkdir -p "${RUN_DIR}"

ray job submit --address="${RAY_DASHBOARD_URL}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "{\"sft\": [1, 0], \"actor\": [1, ${NUM_GPUS}]}" \
   --sft-max-in-flight-steps 4 \
   --num-data-storage-units 8 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${LOG_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "$@" 2>&1 | tee "${RUN_DIR}/train.log"
