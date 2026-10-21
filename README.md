# Xeon 6 vLLM Recipe Validation

Intel-internal CI harness that discovers **all current Xeon 6 renderings** from
`recipes.vllm.ai`, validates each one on a real Xeon 6 host with the official
vLLM CPU image, and runs a quick 128-input / 128-output, concurrency-1
`vllm bench serve` smoke benchmark.

The validation path intentionally follows the upstream Recipes **Getting
Started** flow: `tools/recipes/serve_with_recipe.sh --model ... --hardware
xeon6`. Recipe generation, `vllm serve`, and `vllm bench serve` all run in the
same official Docker container instance for each model.

## Inputs

1. Live Recipes catalog: `https://recipes.vllm.ai/models.json`
2. Per-model Xeon 6 rendering: `/<model>/hw/xeon6.json`
3. Image-bundled `tools/recipes/serve_with_recipe.sh` (optional GitHub override)
4. Intel Xeon 6 self-hosted GitHub runner with Docker access
5. Official CPU image: `vllm/vllm-openai-cpu:latest-x86_64` by default
6. Optional `HF_TOKEN` and a persistent `HF_HOME` cache for gated models

No vLLM installation and no custom image build are required on the runner.

Actions summaries also show the selected image, Recipes URL, additional
container environment, benchmark prompt count, hostname, CPU model/identity,
and topology. Environment input is intended for non-secret settings; use
repository secrets for credentials. Results record override variable names.

For a local run with the same environment input:

```bash
VLLM_TEST_ENV_VARS='VLLM_USE_V2_MODEL_RUNNER=1' ./scripts/weekly.sh
```


## Nightly flow

`.github/workflows/nightly-xeon6.yml` runs the same Xeon 6 validation path every
day using `vllm/vllm-openai-cpu:nightly` by default.

The schedule is `08:00 UTC`, which is approximately **1:00 AM Pacific during
daylight saving time** and **12:00 AM Pacific during standard time**. GitHub
Actions cron schedules are UTC-based, so this intentionally keeps the run near
midnight/1 AM across the year.

Nightly validation publishes to `validated-xeon6-nightly`, keeping nightly
runtime regressions separate from the weekly/stable `validated-xeon6` catalog.
Manual dispatch can override `model`, `vllm_image`, `recipe_api_base`,
`recipe_tools_repo`, `recipe_tools_ref`, and `vllm_env_vars`.

## Weekly flow

```mermaid
flowchart TD
    A["GitHub Actions weekly cron<br/>or workflow_dispatch"] --> B["Intel Xeon 6 self-hosted runner"]

    B --> C["Optional recipe tools checkout"]
    B --> D["Pull official CPU image<br/>vllm/vllm-openai-cpu:latest-x86_64"]
    B --> E["Discover every model with<br/>/hw/xeon6.json"]

    C --> F
    D --> F
    E --> F

    subgraph F["For each Xeon 6 model - sequentially"]
        direction TB

        F1["docker run ONE official CPU-image container<br/><br/>Mounts:<br/>optional tools/recipes → /recipes:ro<br/>HF cache → /hf-cache<br/>work/results → /validation"]
        F2["Image-bundled or override serve_with_recipe.sh<br/>--model MODEL<br/>--hardware xeon6"]
        F3["Automatic Xeon 6 hardware detection<br/>--detect-hardware"]
        F4["Generate config.yml + env.sh"]
        F5["source env.sh"]
        F6["exec vllm serve<br/>--config config.yml"]
        F7{"/health ready?"}
        F8["docker exec SAME container<br/>vllm bench serve<br/><br/>input=128 / output=128<br/>num-prompts=20 weekly / 5 nightly<br/>max-concurrency=1<br/>request-rate=inf<br/>ignore-eos"]
        F9{"Validation result"}

        F1 --> F2 --> F3 --> F4 --> F5 --> F6 --> F7
        F7 -- Yes --> F8 --> F9
        F7 -- No --> F9
    end

    F9 -- PASS --> G["Promote latest known-good bundle<br/>validated/xeon6/org/model/"]
    F9 -- FAIL --> H["Preserve candidate + logs<br/>history/date/run-id/"]

    G --> I["Generate JSON + HTML summary"]
    H --> I
    I --> J["Publish validated-xeon6 branch<br/>validated/ + latest/ + history/"]
```

By default, Docker uses the bundled entrypoint
`/vllm-workspace/tools/recipes/serve_with_recipe.sh`, following the
[official CPU serving instructions](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/#serve-with-vllm-recipes).
No vLLM source checkout is needed. The writable `/validation` working directory
captures generated `config.yml` and `env.sh` for the validation bundle.

When `recipe_tools_repo` is supplied, the harness fetches `recipe_tools_ref` into
a fresh checkout and mounts only `tools/recipes` at `/recipes:ro`, using
`/recipes/serve_with_recipe.sh` as the entrypoint. The vLLM runtime and benchmark
CLI still come from the selected image; this option tests recipe tool changes
without rebuilding or installing vLLM. Changes elsewhere in that repository
are not loaded into the runtime.

`results/recipe-tools.json`, historical reports, and validated bundles record
whether the tools came from the image or GitHub, the override repository/ref
and resolved commit, and the image ID/digests. A bundled run has no separate
tools commit; its provenance is the image ID/digests.

## Final outputs

### Latest known-good files

```text
validated/xeon6/<org>/<model>/
├── config.yml
├── env.sh
├── recipe.json
├── validation.json
├── result.json
├── system-info.txt
└── README.md
```

A failed new run **does not overwrite** the last known-good bundle.

### Weekly results

```text
results/
├── discovery.json
├── docker-image.json
├── recipe-tools.json
├── system-info.txt
├── weekly-summary.json
├── weekly-summary.html
└── models/<org>__<model>/
    ├── config.yml
    ├── env.sh
    ├── recipe.json
    ├── result.json
    ├── server.log
    ├── benchmark.log
    └── benchmark/*.json
```

`server.log` contains both the `serve_with_recipe.sh` generation output and the
vLLM server log because that upstream script generates the files and then
`exec`s the server.

### Historical runs

`history/YYYY-MM-DD/run-<number>-attempt-<attempt>/` keeps the HTML/JSON summary and per-model test evidence for every run, including manual reruns on the same day.

## Persistent publication

The workflow restores the previous `validated/` catalog before testing and,
after the run, publishes the latest known-good bundles to the dedicated
`validated-xeon6` branch. This makes the tested files easy to consume without
searching run-scoped Actions artifacts.

```text
validated-xeon6 branch
├── validated/xeon6/<org>/<model>/config.yml
├── validated/xeon6/<org>/<model>/env.sh
└── latest/xeon6-validation.html
```

The repository workflow needs `contents: write` permission. If Intel policy
does not allow Actions to push branches, disable the publish step and replace
it with your internal artifact/object-storage publication mechanism.

## Runner prerequisites

The self-hosted Xeon runner should have:

- Python 3.10+
- Git
- Docker, with the GitHub runner user allowed to access the Docker daemon
- `numactl` for host provenance collection
- enough persistent model-cache storage for the complete Xeon 6 recipe set
- outbound HTTPS access to GitHub, Recipes, Hugging Face, and Docker Hub

The runner does **not** need host-installed vLLM or `vllm[bench]`.

Register it with labels similar to:

```text
self-hosted, linux, x64, xeon6, recipe-validation
```

Before the first run, verify Docker access as the runner user:

```bash
docker info
docker pull vllm/vllm-openai-cpu:latest-x86_64
```

By default the harness uses `$HOME/.cache/huggingface`. For a larger persistent
runner disk, set the repository variable `HF_HOME` to that path.

## Manual GitHub Actions test

The weekly and nightly workflows expose these manual inputs:

- `model`: exact model ID; leave empty for the full weekly catalog
- `vllm_image`: defaults to `vllm/vllm-openai-cpu:latest-x86_64`
- `recipe_api_base`: defaults to `https://recipes.vllm.ai`; set this to a
  Recipes Vercel preview URL to validate a Recipes PR before merge
- `vllm_env_vars`: optional non-secret `KEY=VALUE` pairs, separated by spaces or
  newlines, applied to every model container. For example:
  `VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_RUST_FRONTEND=1`. Quote values containing
  spaces. Environment values defined by the recipe's generated `env.sh` may
  replace matching container variables when the serving script sources it.
- `recipe_tools_repo`: optional GitHub clone URL, for example
  `https://github.com/intel-ai-tce/vllm.git`; leave empty to use image-bundled tools
- `recipe_tools_ref`: branch, tag, or full commit SHA; defaults to `main` and is
  used only when `recipe_tools_repo` is set

To test the TP detection branch, set `recipe_tools_repo` to
`https://github.com/intel-ai-tce/vllm.git` and `recipe_tools_ref` to
`recipe_tools_TP_fix`. Use the preview deployment in `recipe_api_base` to test
recipe website changes together with those tools. A missing ref or missing
`tools/recipes/serve_with_recipe.sh` fails the run rather than falling back to
the image's scripts.

For the first end-to-end test, trigger **Weekly Xeon 6 Recipe Validation** with
one Xeon6 model. After that passes, leave `model` empty to exercise the same
path across all discovered Xeon6 recipes.

## Local/manual test

Run the complete catalog:

```bash
./scripts/weekly.sh
```

Run one model with the same weekly path:

```bash
TEST_MODEL=meta-llama/Llama-3.1-8B-Instruct ./scripts/weekly.sh
```

Override the official image when needed:

```bash
VLLM_IMAGE=vllm/vllm-openai-cpu:<tag>-x86_64 ./scripts/weekly.sh
```

Validate against a Recipes preview deployment instead of production:

```bash
RECIPES_BASE_URL=https://vllm-recipes-git-fork-intel-ai-tce-xeonvariantsfix-inferact-inc.vercel.app \
  ./scripts/weekly.sh
```

The same `RECIPES_BASE_URL` is used for model discovery and for
`serve_with_recipe.sh --api-base`, so a preview run does not mix preview
discovery with production recipe conversion.

Test a recipe tools fork/branch with the official CPU image:

```bash
RECIPE_TOOLS_REPO=https://github.com/intel-ai-tce/vllm.git \
RECIPE_TOOLS_REF=recipe_tools_TP_fix \
TEST_MODEL=microsoft/Phi-4-reasoning \
  ./scripts/weekly.sh
```

The previous `VLLM_REPO` / `VLLM_REF` environment variables are replaced by the
optional `RECIPE_TOOLS_REPO` / `RECIPE_TOOLS_REF` pair.

## Benchmark policy

The default quick performance check is intentionally small:

- random input length: 128
- requested output length: 128
- prompts: 20 in the weekly workflow; 5 in nightly and local runs
- maximum concurrency: 1
- request rate: infinite (backpressured by max concurrency)
- `--ignore-eos` enabled to keep generated lengths comparable

Performance is recorded but this harness does not impose a TTFT/TPOT regression
threshold. Runtime success/failure and performance trend should remain separate
signals until enough historical data exists to define stable thresholds.

## Specialty/non-generation models

Text-generation models use the default `/v1/completions` + synthetic random
benchmark. Per-model benchmark adapters live in `config/model_overrides.json`.

`openai/whisper-large-v3` uses the OpenAI Audio API instead:

- backend: `openai-audio`
- endpoint: `/v1/audio/transcriptions`
- dataset: `D4nt3/esb-datasets-earnings22-validation-tiny-filtered`
- split: `validation`
- benchmark-only Python dependency: `datasets`

The official serving image intentionally does not need every optional benchmark
dependency. An override may therefore declare `benchmark.python_packages`; the
validator installs those packages into the running container's `/opt/venv` with
`uv` before invoking `vllm bench serve`. This is only for benchmark-client
dependencies and must not be used to patch missing model/runtime dependencies.

Add future embedding, reranker, or other non-generation tasks through the same
override mechanism rather than treating a generic completion-endpoint failure as
a recipe failure.

## KVShrink validation

`.github/workflows/kvshrink-xeon6.yml` (manual dispatch) and
`scripts/kvshrink_validate.sh` measure what
[KVShrink](https://github.com/intel/intel-accel-for-llm) adds to each Xeon 6
recipe. Every model is served twice from the same image, one run after the
other:

| Mode | Server |
| --- | --- |
| `baseline` | the recipe as generated by `serve_with_recipe.sh` |
| `kvshrink` | the same recipe + `KVShrinkConnector` (KV cache compressed on QAT) |

Both runs use the same prefix-hit benchmark: by default 8000 input tokens with
an 80% shared prefix, 128 output tokens, 10 prompts after 5 warmups, max
concurrency 4. vLLM prefix caching is off in both runs
(`KVSHRINK_PREFIX_CACHING=off`), so the baseline recomputes every prompt and
the KVShrink run reuses the prefix only through KVShrink, as in
intel-accel-for-llm's own benchmarks.

Results go to `results/kvshrink-summary.{json,html}` and the Actions step
summary. For each model they list:

- TTFT/TPOT for both runs;
- TTFT speedup, TPOT change and output-throughput ratio;
- the number of requests that loaded cached KV blocks from KVShrink.

A KVShrink run fails when the connector does not register the KV cache or no
request loads blocks from KVShrink. Models whose `config/model_overrides.json`
benchmark is not the random text dataset (for example Whisper) are skipped.

### How the KVShrink run is served

The recipe path is unchanged. The container entrypoint
`scripts/kvshrink/entrypoint.sh` (mounted at `/kvshrink`) only puts the
`scripts/kvshrink/bin/vllm` shim first on `PATH` and runs the upstream
`serve_with_recipe.sh`. The shim passes every `vllm` command through except the
recipe's final `vllm serve --config config.yml`, where it:

1. reads TP/DP from the generated `config.yml` (`kvshrink_layout.py`);
2. gives each rank one NUMA-local CPU per physical core and moves the last
   `KVSHRINK_IAXL_CORES` (default 2) cores of each rank from
   `VLLM_CPU_OMP_THREADS_BIND` to `KVSHRINK_CPU_OMP_THREADS_BIND` for IAXL
   threads;
3. sources intel-accel-for-llm `setvars.sh` with `DEVICE=cpu` (QAT devices
   per rank, IAXL thread settings);
4. adds `--kv-transfer-config` for `KVShrinkConnector`.

In both modes the shim also adds `--no-enable-prefix-caching` when prefix
caching is off. `setvars.sh` sets the host `kernel.numa_balancing=0` from the
privileged container, as the intel-accel-for-llm dev container does.

### KVShrink image

`scripts/build_kvshrink_image.sh` builds intel-accel-for-llm
`docker/Dockerfile.dev` (`DEVICE=cpu`) on `VLLM_IMAGE`, then
`docker/Dockerfile.kvshrink` installs IAXL/KVShrink from the same source into
`/opt/intel-accel-for-llm`. The tag is derived from the base image ID and the
source contents, so unchanged inputs reuse the Docker cache. Source selection:

- `KVSHRINK_IMAGE`: use a prebuilt image (it must provide `iaxl`, `kvshrink`
  and `$IAXL_SRC/setvars.sh`) and skip the build;
- `KVSHRINK_SRC`: a local checkout; its tracked files are used, including
  uncommitted edits;
- `KVSHRINK_REPO` + `KVSHRINK_REF`: clone and build.

`IAXL_QATLIB=intree` (default) builds qatlib for the in-tree QAT driver and
mounts the host `/run/qat` so all ranks share QAT through the host `qatmgr`.
Set `QATLIB_VERSION` to the host qatlib version when it differs from the
intel-accel-for-llm default. `results/kvshrink-image.json` records the image,
base image, and KVShrink source/commit.

### Running

The runner needs QAT (VFs bound to `vfio-pci` and a running `qatmgr` for
`intree`) and the `qat` runner label in addition to the Xeon 6 labels. The
workflow checks out `kvshrink_repository` at `kvshrink_ref`. Set the
`KVSHRINK_REPO_TOKEN` secret if that repository is private.

```bash
# Build from a local checkout and validate one model
KVSHRINK_SRC=~/intel-accel-for-llm TEST_MODEL=Qwen/Qwen3-8B ./scripts/kvshrink_validate.sh

# Reuse a prebuilt image; TEST_MODEL takes several space-separated models
KVSHRINK_IMAGE=vllm-recipes-kvshrink:<tag> \
TEST_MODEL="Qwen/Qwen3-8B meta-llama/Llama-3.1-8B-Instruct" ./scripts/kvshrink_validate.sh
```

`config/kvshrink.env` holds the defaults:

- workload: `KVSHRINK_INPUT_LEN`, `KVSHRINK_HIT_RATE`, `KVSHRINK_OUTPUT_LEN`,
  `KVSHRINK_NUM_PROMPTS`, `KVSHRINK_NUM_WARMUPS`, `KVSHRINK_MAX_CONCURRENCY`;
- `KVSHRINK_DOCKER_ARGS`: KVShrink-run container options, privileged with
  `/dev` by default for QAT;
- `KVSHRINK_ENV_VARS`: `KEY=VALUE` IAXL settings for the KVShrink run,
  `IAXL_DDR_POOL_SIZE_GB=16 IAXL_CPU_ZIP_ENABLE=0` by default. To use IAA
  instead of QAT, for example, set
  `KVSHRINK_ENV_VARS='IAXL_QAT_ZIP_ENABLE=0 IAXL_IAA_ZIP_ENABLE=1 IAXL_DDR_POOL_SIZE_GB=16'`.

Running every Xeon 6 recipe takes much longer than the weekly validation
because each model is served twice with 8000-token prompts.
