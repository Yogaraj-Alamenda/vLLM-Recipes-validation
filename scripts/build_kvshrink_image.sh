#!/usr/bin/env bash
# Produce the vLLM CPU image used by the KVShrink validation.
#
# Usage: build_kvshrink_image.sh <metadata.json>
#
# Prints the image reference on the last stdout line; build logs go to stderr.
#
# Inputs (environment):
#   VLLM_IMAGE       vLLM CPU image to build on (required unless KVSHRINK_IMAGE is set)
#   KVSHRINK_IMAGE   prebuilt image with IAXL/KVShrink installed; skips the build
#   KVSHRINK_SRC     local intel-accel-for-llm checkout (tracked files are used,
#                    including uncommitted edits)
#   KVSHRINK_REPO    intel-accel-for-llm clone URL, used when KVSHRINK_SRC is empty
#   KVSHRINK_REF     branch, tag, or commit for KVSHRINK_REPO (default: main)
#   IAXL_QATLIB      intree (in-tree QAT driver + host qatmgr, default) or empty
#                    for the out-of-tree QAT package
#   QATLIB_VERSION   qatlib built into the image for IAXL_QATLIB=intree; match
#                    the host qatmgr (default: the Dockerfile.dev default)
#   WORK_ROOT        scratch directory for the build context
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
METADATA_OUT=${1:?usage: build_kvshrink_image.sh <metadata.json>}

KVSHRINK_IMAGE=${KVSHRINK_IMAGE:-}
KVSHRINK_SRC=${KVSHRINK_SRC:-}
KVSHRINK_REPO=${KVSHRINK_REPO:-}
KVSHRINK_REF=${KVSHRINK_REF:-main}
IAXL_QATLIB=${IAXL_QATLIB-intree}
QATLIB_VERSION=${QATLIB_VERSION:-}
WORK_ROOT=${WORK_ROOT:-${GITHUB_WORKSPACE:-$ROOT}/work}

log() { echo "$@" >&2; }

write_metadata() {
  local image=$1 source=$2 repo=$3 ref=$4 commit=$5 dirty=$6
  python3 - "$METADATA_OUT" "$image" "$source" "$repo" "$ref" "$commit" "$dirty" \
    "${VLLM_IMAGE:-}" "$IAXL_QATLIB" "$QATLIB_VERSION" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

out, image, source, repo, ref, commit, dirty, base, qatlib, qatlib_version = sys.argv[1:]


def inspect(ref):
    if not ref:
        return {}
    proc = subprocess.run(
        ["docker", "image", "inspect", ref], capture_output=True, text=True
    )
    if proc.returncode:
        return {}
    info = json.loads(proc.stdout)[0]
    return {"id": info.get("Id"), "digests": info.get("RepoDigests", [])}


img = inspect(image)
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps({
    "image": image,
    "image_id": img.get("id"),
    "image_digests": img.get("digests", []),
    "source": source,
    "repository": repo or None,
    "ref": ref or None,
    "commit": commit or None,
    "dirty": dirty == "1",
    "base_image": base or None,
    "base_image_id": inspect(base).get("id") if source != "prebuilt" else None,
    "iaxl_qatlib": qatlib or "oot",
    "qatlib_version": qatlib_version or None,
}, indent=2) + "\n")
PY
}

if [[ -n "$KVSHRINK_IMAGE" ]]; then
  log "==> Using prebuilt KVShrink image: $KVSHRINK_IMAGE"
  docker image inspect "$KVSHRINK_IMAGE" >/dev/null 2>&1 || docker pull "$KVSHRINK_IMAGE" >&2
  if ! docker run --rm --entrypoint python3 "$KVSHRINK_IMAGE" \
      -c 'import os, iaxl, kvshrink.kvshrink_connector; assert os.path.isfile(os.path.join(os.environ["IAXL_SRC"], "setvars.sh"))' >&2; then
    log "ERROR: $KVSHRINK_IMAGE must provide iaxl, kvshrink and \$IAXL_SRC/setvars.sh"
    exit 2
  fi
  write_metadata "$KVSHRINK_IMAGE" prebuilt "" "" "" 0
  echo "$KVSHRINK_IMAGE"
  exit 0
fi

: "${VLLM_IMAGE:?VLLM_IMAGE is required to build the KVShrink image}"
mkdir -p "$WORK_ROOT"
SCRATCH=$(mktemp -d "$WORK_ROOT/kvshrink-build.XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT

if [[ -n "$KVSHRINK_SRC" ]]; then
  SRC=$(cd "$KVSHRINK_SRC" && pwd)
  SOURCE=local
  REPO=$(git -C "$SRC" remote get-url origin 2>/dev/null || true)
  REF=$(git -C "$SRC" rev-parse --abbrev-ref HEAD)
elif [[ -n "$KVSHRINK_REPO" ]]; then
  SRC="$SCRATCH/src"
  SOURCE=github
  REPO=$KVSHRINK_REPO
  REF=$KVSHRINK_REF
  log "==> Fetching $REPO @ $REF"
  git init -q "$SRC"
  git -C "$SRC" remote add origin "$REPO"
  git -C "$SRC" fetch -q --depth 1 origin "$REF"
  git -C "$SRC" checkout -q --detach FETCH_HEAD
else
  log "ERROR: set KVSHRINK_IMAGE, KVSHRINK_SRC, or KVSHRINK_REPO"
  exit 2
fi
COMMIT=$(git -C "$SRC" rev-parse HEAD)
DIRTY=0
[[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]] || DIRTY=1

for f in docker/Dockerfile.dev setvars.sh setup.py; do
  if [[ ! -f "$SRC/$f" ]]; then
    log "ERROR: $SRC is not an intel-accel-for-llm checkout (missing $f)"
    exit 2
  fi
done

# Only tracked files (with their working-tree contents) go into the context, so
# local build outputs, model caches and documents never reach the image.
CTX="$SCRATCH/context"
mkdir -p "$CTX"
git -C "$SRC" ls-files -z | tar -C "$SRC" --null --ignore-failed-read -T - -cf - | tar -C "$CTX" -xf -
CONTENT_HASH=$(cd "$CTX" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)

docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1 || docker pull "$VLLM_IMAGE" >&2
BASE_ID=$(docker image inspect --format '{{.Id}}' "$VLLM_IMAGE" | sed 's/^sha256://' | cut -c1-12)
TAG="$BASE_ID-$CONTENT_HASH${IAXL_QATLIB:+-$IAXL_QATLIB}"
DEV_IMAGE="vllm-recipes-kvshrink-dev:$TAG"
IMAGE="vllm-recipes-kvshrink:$TAG"

QATLIB_ARGS=()
[[ -n "$QATLIB_VERSION" ]] && QATLIB_ARGS=(--build-arg "QATLIB_VERSION=$QATLIB_VERSION")

# Runners often export only one spelling of the proxy variables; apt, git and
# pip inside the build need them, so forward both spellings.
PROXY_ARGS=()
for name in http_proxy https_proxy no_proxy; do
  upper=${name^^}
  value=${!name:-${!upper:-}}
  if [[ -n "$value" ]]; then
    PROXY_ARGS+=(--build-arg "$name=$value" --build-arg "$upper=$value")
  fi
done

log "==> Building $DEV_IMAGE from $SRC ($COMMIT, dirty=$DIRTY) on $VLLM_IMAGE"
docker build -f "$CTX/docker/Dockerfile.dev" -t "$DEV_IMAGE" \
  --build-arg "BASE=$VLLM_IMAGE" \
  --build-arg DEVICE=cpu \
  --build-arg "IAXL_QATLIB=$IAXL_QATLIB" \
  "${QATLIB_ARGS[@]}" \
  "${PROXY_ARGS[@]}" \
  "$CTX" >&2

log "==> Installing IAXL/KVShrink into $IMAGE"
docker build -f "$ROOT/docker/Dockerfile.kvshrink" -t "$IMAGE" \
  --build-arg "BASE=$DEV_IMAGE" \
  --build-arg "IAXL_QATLIB=$IAXL_QATLIB" \
  "${PROXY_ARGS[@]}" \
  "$CTX" >&2

write_metadata "$IMAGE" "$SOURCE" "$REPO" "$REF" "$COMMIT" "$DIRTY"
echo "$IMAGE"
