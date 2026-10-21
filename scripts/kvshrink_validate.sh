#!/usr/bin/env bash
# KVShrink validation on Xeon 6 recipes: for every model, serve the recipe as
# generated (baseline) and with KVShrink, run the same prefix-hit benchmark on
# both, and report the TTFT/TPOT deltas.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "$ROOT/config/settings.env"
# shellcheck disable=SC1091
source "$ROOT/config/kvshrink.env"
export KVSHRINK_IMAGE KVSHRINK_SRC KVSHRINK_REPO KVSHRINK_REF IAXL_QATLIB QATLIB_VERSION
export KVSHRINK_PREFIX_CACHING KVSHRINK_IAXL_CORES KVSHRINK_DOCKER_ARGS KVSHRINK_ENV_VARS
export WORK_ROOT

RUN_DATE=$(date -u +%F)
RUN_ID=${VALIDATION_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}

mkdir -p "$WORK_ROOT" "$HF_HOME"

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is not available to the GitHub runner user." >&2
  exit 2
fi

rm -rf "$RESULT_ROOT"
mkdir -p "$RESULT_ROOT"
"$ROOT/scripts/collect_system_info.sh" "$RESULT_ROOT/system-info.txt"

if [[ -z "$KVSHRINK_IMAGE" ]]; then
  echo "==> Pulling vLLM CPU image: $VLLM_IMAGE"
  docker pull "$VLLM_IMAGE"
fi
echo "==> Preparing KVShrink image"
IMAGE=$("$ROOT/scripts/build_kvshrink_image.sh" "$RESULT_ROOT/kvshrink-image.json" | tail -n 1)
echo "==> KVShrink image: $IMAGE"
docker image inspect "$IMAGE" > "$RESULT_ROOT/docker-image.json"

TOOLS_ARGS=()
TOOLS_DIR=""
TOOLS_SHA=""
if [[ -n "$RECIPE_TOOLS_REPO" ]]; then
  TOOLS_DIR=$(mktemp -d "$WORK_ROOT/recipe-tools.XXXXXX")
  trap 'rm -rf "$TOOLS_DIR"' EXIT
  git -C "$TOOLS_DIR" init -q
  git -C "$TOOLS_DIR" remote add origin "$RECIPE_TOOLS_REPO"
  git -C "$TOOLS_DIR" fetch --depth 1 origin "$RECIPE_TOOLS_REF"
  git -C "$TOOLS_DIR" checkout -q --detach FETCH_HEAD
  TOOLS_SHA=$(git -C "$TOOLS_DIR" rev-parse HEAD)
  if [[ ! -f "$TOOLS_DIR/tools/recipes/serve_with_recipe.sh" ]]; then
    echo "ERROR: tools/recipes/serve_with_recipe.sh not found in requested source" >&2
    exit 2
  fi
  TOOLS_ARGS+=(--vllm-dir "$TOOLS_DIR")
elif ! docker run --rm --entrypoint /usr/bin/test "$IMAGE" \
    -x /vllm-workspace/tools/recipes/serve_with_recipe.sh; then
  echo "ERROR: $IMAGE has no /vllm-workspace/tools/recipes/serve_with_recipe.sh;" \
    "use a newer VLLM_IMAGE or set RECIPE_TOOLS_REPO/RECIPE_TOOLS_REF" >&2
  exit 2
fi

python3 - "$RESULT_ROOT/recipe-tools.json" "$RECIPE_TOOLS_REPO" "$RECIPE_TOOLS_REF" "$TOOLS_SHA" "$IMAGE" <<'PY'
import json
import sys
from pathlib import Path

out, repo, ref, sha, image = sys.argv[1:]
Path(out).write_text(json.dumps({
    "source": "github" if repo else "image",
    "repository": repo or None,
    "ref": ref if repo else None,
    "commit": sha or None,
    "image": image,
}, indent=2) + "\n")
PY

python3 "$ROOT/scripts/discover_xeon6_models.py" \
  --base-url "$RECIPES_BASE_URL" \
  --output "$RESULT_ROOT/discovery.json"

ONLY_ARGS=()
for model in $TEST_MODEL; do
  ONLY_ARGS+=(--only "$model")
done

set +e
python3 "$ROOT/scripts/run_kvshrink_validation.py" \
  --discovery "$RESULT_ROOT/discovery.json" \
  "${TOOLS_ARGS[@]}" \
  --image "$IMAGE" \
  --recipes-base-url "$RECIPES_BASE_URL" \
  --hf-home "$HF_HOME" \
  --work-root "$WORK_ROOT/kvshrink" \
  --result-root "$RESULT_ROOT/models" \
  --hardware "$HARDWARE" \
  --input-len "$KVSHRINK_INPUT_LEN" \
  --hit-rate "$KVSHRINK_HIT_RATE" \
  --output-len "$KVSHRINK_OUTPUT_LEN" \
  --num-prompts "$KVSHRINK_NUM_PROMPTS" \
  --num-warmups "$KVSHRINK_NUM_WARMUPS" \
  --max-concurrency "$KVSHRINK_MAX_CONCURRENCY" \
  --server-timeout "$SERVER_START_TIMEOUT_SEC" \
  --bench-timeout "$KVSHRINK_BENCH_TIMEOUT_SEC" \
  "${ONLY_ARGS[@]}"
RUN_RC=$?
set -e

python3 "$ROOT/scripts/kvshrink_report.py" \
  --results "$RESULT_ROOT/models" \
  --image-metadata "$RESULT_ROOT/kvshrink-image.json" \
  --run-id "$RUN_ID" \
  --run-date "$RUN_DATE" \
  --prefix-caching "$KVSHRINK_PREFIX_CACHING"
cp "$RESULT_ROOT/models/kvshrink-summary.json" "$RESULT_ROOT/kvshrink-summary.json"
cp "$RESULT_ROOT/models/kvshrink-summary.html" "$RESULT_ROOT/kvshrink-summary.html"

exit "$RUN_RC"
