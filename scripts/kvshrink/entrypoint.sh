#!/usr/bin/env bash
# Run a command with the KVShrink vllm shim first on PATH.
#
# The validation container uses this as its entrypoint in front of the
# upstream serve_with_recipe.sh, so recipe conversion is untouched and only
# its final `exec vllm serve --config config.yml` reaches bin/vllm.
set -euo pipefail

KVSHRINK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PATH="$KVSHRINK_DIR/bin:$PATH"
exec "$@"
