#!/usr/bin/env bash
# Reuse the nightly recipe/inference harness, but keep all diagnostics isolated.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../config/settings.env
source "$ROOT/config/settings.env"

# The existing runner mounts RESULT_ROOT/models/<model> as /results, so each
# model gets an isolated Triton dump and the server process can write to it.
export TRITON_TRACE_SITE_DIR="$ROOT/scripts"
export TRITON_NATIVE_PERF_SCRIPT="$ROOT/scripts/native_cpu_perf.py"
export VLLM_TEST_ENV_VARS="${VLLM_TEST_ENV_VARS:-} VLLM_USE_V2_MODEL_RUNNER=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/results/triton TRITON_RUNTIME_TRACE=1 TRITON_RUNTIME_TRACE_OUTPUT=/results/triton-launches.jsonl"
RUN_DATE=$(date -u +%F)
RUN_ID=${VALIDATION_RUN_ID:-triton-local-$(date -u +%Y%m%dT%H%M%SZ)}
INSPECTION_ROOT="$ROOT/triton-results"
RESULT_MODELS="$INSPECTION_ROOT/models"
mkdir -p "$WORK_ROOT" "$HF_HOME"
rm -rf "$INSPECTION_ROOT"
mkdir -p "$RESULT_MODELS"

docker info >/dev/null
# Track the exact resolved image for reproducibility.
docker pull "$VLLM_IMAGE"
docker image inspect "$VLLM_IMAGE" > "$INSPECTION_ROOT/docker-image.json"
"$ROOT/scripts/collect_system_info.sh" "$INSPECTION_ROOT/system-info.txt"

TOOLS_ARGS=()
TOOLS_SHA=""
TOOLS_DIR=""
if [[ -n "$RECIPE_TOOLS_REPO" ]]; then
  TOOLS_DIR=$(mktemp -d "$WORK_ROOT/triton-recipe-tools.XXXXXX")
  trap 'rm -rf "$TOOLS_DIR"' EXIT
  git -C "$TOOLS_DIR" init -q
  git -C "$TOOLS_DIR" remote add origin "$RECIPE_TOOLS_REPO"
  git -C "$TOOLS_DIR" fetch --depth 1 origin "$RECIPE_TOOLS_REF"
  git -C "$TOOLS_DIR" checkout -q --detach FETCH_HEAD
  TOOLS_SHA=$(git -C "$TOOLS_DIR" rev-parse HEAD)
  test -f "$TOOLS_DIR/tools/recipes/serve_with_recipe.sh"
  TOOLS_ARGS=(--vllm-dir "$TOOLS_DIR")
else
  docker run --rm --entrypoint /usr/bin/test "$VLLM_IMAGE" \
    -x /vllm-workspace/tools/recipes/serve_with_recipe.sh
fi
python3 - "$INSPECTION_ROOT/recipe-tools.json" "$RECIPE_TOOLS_REPO" "$RECIPE_TOOLS_REF" "$TOOLS_SHA" <<'PY'
import json,sys
from pathlib import Path
out,repo,ref,sha=sys.argv[1:]
Path(out).write_text(json.dumps({'source':'github' if repo else 'image',
 'repository':repo or None,'ref':ref if repo else None,'commit':sha or None},indent=2)+'\n')
PY
python3 "$ROOT/scripts/discover_xeon6_models.py" \
  --base-url "$RECIPES_BASE_URL" --output "$INSPECTION_ROOT/discovery.json"
ONLY_ARGS=()
if [[ -n "$TEST_MODEL" ]]; then ONLY_ARGS=(--only "$TEST_MODEL"); fi
# This is a low-load inference probe, NOT a performance benchmark. Retain the
# original per-model endpoint/dataset overrides (including multimodal cases).
set +e
python3 "$ROOT/scripts/run_all.py" \
  --discovery "$INSPECTION_ROOT/discovery.json" \
  "${TOOLS_ARGS[@]}" --image "$VLLM_IMAGE" \
  --recipes-base-url "$RECIPES_BASE_URL" --hf-home "$HF_HOME" \
  --work-root "$WORK_ROOT/triton-models" --result-root "$RESULT_MODELS" \
  --hardware "$HARDWARE" --input-len 128 --output-len 128 \
  --num-prompts "${TRITON_PROBE_PROMPTS:-1}" --max-concurrency 1 \
  --request-rate inf --server-timeout "$SERVER_START_TIMEOUT_SEC" \
  --bench-timeout "$BENCH_TIMEOUT_SEC" "${ONLY_ARGS[@]}"
RUN_RC=$?
set -e
python3 "$ROOT/scripts/triton_inspect_report.py" \
  --results "$RESULT_MODELS" --output "$INSPECTION_ROOT" \
  --run-id "$RUN_ID" --run-date "$RUN_DATE" --image "$VLLM_IMAGE"
exit "$RUN_RC"
