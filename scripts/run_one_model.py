#!/usr/bin/env python3
"""Validate one Xeon 6 recipe in one official vLLM CPU container."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(
    cmd,
    *,
    cwd=None,
    env=None,
    stdout=None,
    stderr=None,
    timeout=None,
    check=True,
    capture_output=False,
):
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        timeout=timeout,
        check=check,
        text=True,
        capture_output=capture_output,
    )


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_container_env_overrides(raw: str) -> list[tuple[str, str]]:
    """Parse optional run-wide KEY=VALUE entries for docker run."""
    if not raw.strip():
        return []

    parsed: dict[str, str] = {}
    for item in shlex.split(raw):
        if "=" not in item:
            raise ValueError(
                "container environment override must use KEY=VALUE syntax: "
                f"{item!r}"
            )
        name, value = item.split("=", 1)
        if not _ENV_NAME_RE.fullmatch(name):
            raise ValueError(
                f"invalid container environment variable name: {name!r}"
            )
        parsed[name] = value

    return list(parsed.items())


def docker_container_running(name: str) -> bool:
    try:
        proc = run(
            ["docker", "inspect", "--format", "{{.State.Running}}", name],
            timeout=10,
            capture_output=True,
        )
        return proc.stdout.strip().lower() == "true"
    except Exception:
        return False


def docker_http_ok(name: str, url: str, timeout: int = 5) -> bool:
    try:
        run(
            [
                "docker",
                "exec",
                name,
                "curl",
                "-fsS",
                "--max-time",
                str(timeout),
                url,
            ],
            timeout=timeout + 5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def get_served_model_name(
    name: str, base_url: str, requested_model: str, timeout: int = 5
) -> str:
    """Resolve the model name accepted by the running OpenAI API."""
    proc = run(
        [
            "docker",
            "exec",
            name,
            "curl",
            "-fsS",
            "--max-time",
            str(timeout),
            f"{base_url}/v1/models",
        ],
        timeout=timeout + 5,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    model_ids = [
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if not model_ids:
        raise RuntimeError("/v1/models returned no model IDs")

    if requested_model in model_ids:
        return requested_model

    requested_lower = requested_model.lower()
    case_insensitive = [m for m in model_ids if m.lower() == requested_lower]
    if len(case_insensitive) == 1:
        return case_insensitive[0]

    if len(model_ids) == 1:
        return model_ids[0]

    raise RuntimeError(
        "unable to identify served model for "
        f"{requested_model!r}; /v1/models returned {model_ids!r}"
    )


def wait_ready(name: str, base_url: str, timeout_sec: int) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if not docker_container_running(name):
            return False
        if docker_http_ok(name, f"{base_url}/health"):
            return True
        time.sleep(5)
    return False


def save_docker_logs(name: str, output: Path) -> None:
    try:
        proc = run(
            ["docker", "logs", name],
            timeout=60,
            capture_output=True,
            check=False,
        )
        output.write_text((proc.stdout or "") + (proc.stderr or ""))
    except Exception as exc:
        output.write_text(f"Unable to collect docker logs: {exc}\n")


def remove_container(name: str) -> None:
    run(
        ["docker", "rm", "-f", name],
        timeout=60,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def image_metadata(image: str) -> dict:
    metadata = {"image": image}
    try:
        proc = run(
            ["docker", "image", "inspect", image],
            timeout=30,
            capture_output=True,
        )
        inspected = json.loads(proc.stdout)[0]
        metadata["image_id"] = inspected.get("Id")
        metadata["repo_digests"] = inspected.get("RepoDigests", [])
    except Exception as exc:
        metadata["inspect_error"] = str(exc)
    return metadata


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_model_override(model: str) -> dict:
    """Load an optional per-model validation override."""
    path = Path(__file__).resolve().parents[1] / "config/model_overrides.json"
    if not path.is_file():
        return {}
    data = load_json(path)
    models = data.get("models", {})
    if not isinstance(models, dict):
        raise ValueError(f"invalid model overrides file: {path}")
    override = models.get(model, {})
    if not isinstance(override, dict):
        raise ValueError(f"invalid override for {model!r}: expected an object")
    return override


def find_latest_json(directory: Path):
    files = sorted(
        directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return files[0] if files else None


def extract_metrics(data):
    # vllm bench serve output fields have evolved. Preserve the complete raw JSON
    # and normalize the common metrics when present.
    aliases = {
        "completed": ["completed", "completed_requests", "num_completed_requests"],
        "request_throughput": ["request_throughput", "request_throughput_rps"],
        "output_throughput": ["output_throughput", "output_throughput_tps"],
        "mean_ttft_ms": ["mean_ttft_ms", "mean_ttft"],
        "median_ttft_ms": ["median_ttft_ms", "median_ttft"],
        "p99_ttft_ms": ["p99_ttft_ms", "p99_ttft"],
        "mean_tpot_ms": ["mean_tpot_ms", "mean_tpot"],
        "median_tpot_ms": ["median_tpot_ms", "median_tpot"],
        "p99_tpot_ms": ["p99_tpot_ms", "p99_tpot"],
        "mean_e2el_ms": ["mean_e2el_ms", "mean_e2el", "mean_e2e_latency_ms"],
    }
    out = {}
    for normalized, keys in aliases.items():
        for key in keys:
            if key in data:
                out[normalized] = data[key]
                break
    return out


def copy_candidates(work: Path, result_dir: Path) -> None:
    for name in ("recipe.json", "config.yml", "env.sh"):
        src = work / name
        if src.exists():
            shutil.copy2(src, result_dir / name)


def write_result(path: Path, result: dict) -> None:
    result["finished_at"] = utc_now()
    path.write_text(json.dumps(result, indent=2) + "\n")


def kvshrink_docker_args(args, result: dict) -> list[str]:
    """Container options for one KVShrink validation mode (mount, mode, devices, env)."""
    shim_dir = Path(args.kvshrink_dir).resolve()
    for name in ("entrypoint.sh", "bin/vllm", "kvshrink_layout.py"):
        if not (shim_dir / name).is_file():
            raise FileNotFoundError(f"KVShrink shim file missing: {shim_dir / name}")

    prefix_caching = os.environ.get("KVSHRINK_PREFIX_CACHING", "off")
    if prefix_caching not in ("on", "off"):
        raise ValueError("KVSHRINK_PREFIX_CACHING must be on or off")

    docker_args = [
        "-v", f"{shim_dir}:/kvshrink:ro",
        "-e", f"KVSHRINK_MODE={args.kvshrink_mode}",
        "-e", f"KVSHRINK_PREFIX_CACHING={prefix_caching}",
    ]
    env_names: list[str] = []
    if args.kvshrink_mode == "kvshrink":
        docker_args.extend(shlex.split(os.environ.get("KVSHRINK_DOCKER_ARGS", "")))
        if os.environ.get("KVSHRINK_IAXL_CORES"):
            docker_args.extend(
                ["-e", f"KVSHRINK_IAXL_CORES={os.environ['KVSHRINK_IAXL_CORES']}"]
            )
        # IAXL_QATLIB=intree ranks share QAT VFs through the host qatmgr.
        if Path("/run/qat/qatmgr.sock").is_socket():
            docker_args.extend(["-v", "/run/qat:/run/qat"])
        for name, value in parse_container_env_overrides(
            os.environ.get("KVSHRINK_ENV_VARS", "")
        ):
            docker_args.extend(["-e", f"{name}={value}"])
            env_names.append(name)

    result["kvshrink"] = {
        "mode": args.kvshrink_mode,
        "prefix_caching": prefix_caching,
        "env_override_names": env_names,
    }
    return docker_args


_KVSHRINK_HIT_RE = re.compile(r"externally-cached tokens: (\d+)")


def kvshrink_log_stats(server_log: Path) -> dict:
    """Summarize KVShrink activity from the connector log lines."""
    text = server_log.read_text(errors="replace") if server_log.is_file() else ""
    hits = [int(m) for m in _KVSHRINK_HIT_RE.findall(text)]
    layout = next(
        (
            line.strip()
            for line in text.splitlines()
            if line.startswith("==> KVShrink rank layout")
        ),
        None,
    )
    return {
        "connector_registered": "KV cache layers with shape" in text,
        "lookups": len(hits),
        "hit_requests": sum(1 for n in hits if n > 0),
        "external_tokens": sum(hits),
        "rank_layout": layout,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--recipe-url", required=True)
    ap.add_argument(
        "--vllm-dir",
        help="Optional checkout supplying tools/recipes; default: image-bundled tools",
    )
    ap.add_argument("--image", required=True)
    ap.add_argument("--recipes-base-url", default="https://recipes.vllm.ai")
    ap.add_argument("--hf-home", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--hardware", default="xeon6")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-timeout", type=int, default=3600)
    ap.add_argument("--bench-timeout", type=int, default=1800)
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--max-concurrency", type=int, default=1)
    ap.add_argument("--request-rate", default="inf")
    ap.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="Shared random prefix tokens per prompt (adds --random-prefix-len)",
    )
    ap.add_argument("--num-warmups", type=int, default=0)
    ap.add_argument(
        "--kvshrink-mode",
        choices=("baseline", "kvshrink"),
        help="Run one KVShrink validation mode through the scripts/kvshrink vllm shim",
    )
    ap.add_argument(
        "--kvshrink-dir",
        default=str(Path(__file__).resolve().parent / "kvshrink"),
        help="Directory with the KVShrink entrypoint and vllm shim",
    )
    args = ap.parse_args()

    safe_name = args.model.replace("/", "__")
    work = (Path(args.work_root) / safe_name).resolve()
    result_dir = (Path(args.result_root) / safe_name).resolve()
    recipes_dir = (
        (Path(args.vllm_dir) / "tools/recipes").resolve() if args.vllm_dir else None
    )
    hf_home = Path(args.hf_home).resolve()

    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    result = {
        "model": args.model,
        "hardware": args.hardware,
        "started_at": utc_now(),
        "recipe_url": args.recipe_url,
        "recipes_base_url": args.recipes_base_url,
        "runtime": image_metadata(args.image),
        "workload": {
            "input_tokens": args.input_len,
            "output_tokens": args.output_len,
            "num_prompts": args.num_prompts,
            "max_concurrency": args.max_concurrency,
            "request_rate": args.request_rate,
            "prefix_tokens": args.prefix_len,
            "num_warmups": args.num_warmups,
        },
        "stages": {},
        "status": "FAIL",
    }
    result_path = result_dir / "result.json"

    # Save the live rendering next to the generated files for provenance. The
    # actual generation path below intentionally uses serve_with_recipe.sh with
    # --model/--hardware, matching the documented Getting Started flow.
    recipe_path = work / "recipe.json"
    try:
        req = urllib.request.Request(
            args.recipe_url,
            headers={"User-Agent": "xeon-recipes-weekly-ci/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            recipe_path.write_bytes(response.read())
        result["stages"]["recipe_fetch"] = "PASS"
    except Exception as exc:
        result["stages"]["recipe_fetch"] = "FAIL"
        result["error"] = f"recipe_fetch: {exc}"
        write_result(result_path, result)
        return 1

    if (
        recipes_dir is not None
        and not recipes_dir.joinpath("serve_with_recipe.sh").is_file()
    ):
        result["stages"]["conversion"] = "FAIL"
        result["error"] = f"serve_with_recipe.sh not found under {recipes_dir}"
        copy_candidates(work, result_dir)
        write_result(result_path, result)
        return 1

    # Keep names short and deterministic while avoiding collisions with stale
    # containers from another model.
    model_hash = hashlib.sha1(args.model.encode()).hexdigest()[:12]
    container = f"xeon-recipe-{model_hash}-{os.getpid()}"
    base_url = f"http://{args.host}:{args.port}"
    server_log_path = result_dir / "server.log"

    try:
        container_env_overrides = parse_container_env_overrides(
            os.environ.get("VLLM_TEST_ENV_VARS", "")
        )
        result["stages"]["container_env"] = (
            "PASS" if container_env_overrides else "SKIP"
        )
        # Record names only; values may be sensitive and are already represented
        # in the workflow summary with basic masking for secret-like names.
        result["runtime"]["container_env_override_names"] = [
            name for name, _ in container_env_overrides
        ]
    except ValueError as exc:
        result["stages"]["container_env"] = "FAIL"
        result["error"] = f"container_env: {exc}"
        write_result(result_path, result)
        return 1

    serve_script = (
        "/recipes/serve_with_recipe.sh"
        if recipes_dir
        else "/vllm-workspace/tools/recipes/serve_with_recipe.sh"
    )
    kvshrink_args: list[str] = []
    if args.kvshrink_mode:
        try:
            kvshrink_args = kvshrink_docker_args(args, result)
        except (OSError, ValueError) as exc:
            result["stages"]["container_env"] = "FAIL"
            result["error"] = f"kvshrink_env: {exc}"
            write_result(result_path, result)
            return 1

    docker_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container,
        "--entrypoint",
        "/kvshrink/entrypoint.sh" if args.kvshrink_mode else serve_script,
        "--security-opt",
        "seccomp=unconfined",
        "--cap-add",
        "SYS_NICE",
        "--shm-size=4g",
        "-v",
        f"{work}:/validation",
        "-v",
        f"{result_dir}:/results",
        "-v",
        f"{hf_home}:/hf-cache",
        "-e",
        "HF_HOME=/hf-cache",
        "-e",
        "HF_HUB_CACHE=/hf-cache/hub",
        "-w",
        "/validation",
    ]

    # Optional Triton runtime tracing for the dedicated inspection workflow.
    # sitecustomize is loaded in each Python interpreter (including spawned
    # engine workers), before importing vLLM or Triton. Other CI is unchanged.
    trace_site_dir = os.environ.get("TRITON_TRACE_SITE_DIR", "")
    if trace_site_dir:
        trace_module = (Path(trace_site_dir) / "triton_runtime_sitecustomize.py").resolve()
        if not trace_module.is_file():
            raise FileNotFoundError(f"Triton runtime trace module missing: {trace_module}")
        image_info = json.loads(run(
            ["docker", "image", "inspect", args.image],
            capture_output=True, timeout=30,
        ).stdout)[0]
        env_entries = image_info.get("Config", {}).get("Env") or []
        image_pythonpath = next((e.split("=", 1)[1] for e in env_entries
                                 if e.startswith("PYTHONPATH=")), "")
        docker_cmd.extend(["-v", f"{trace_module}:/triton-trace-site/sitecustomize.py:ro",
                           "-e", f"PYTHONPATH=/triton-trace-site:{image_pythonpath}"])

    if recipes_dir is not None:
        docker_cmd.extend(["-v", f"{recipes_dir}:/recipes:ro"])

    # Preserve credentials/network settings needed by corporate runners without
    # writing secret values into the command line or result files.
    inherited_env = (
        "HF_TOKEN",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
    )
    for name in inherited_env:
        if os.environ.get(name):
            docker_cmd.extend(["-e", name])

    # Apply workflow-provided run-wide overrides to every model container.
    # docker exec inherits these variables for the benchmark process as well.
    for name, value in container_env_overrides:
        docker_cmd.extend(["-e", f"{name}={value}"])

    docker_cmd.extend(kvshrink_args)
    docker_cmd.append(args.image)
    if args.kvshrink_mode:
        docker_cmd.append(serve_script)

    docker_cmd.extend(
        [
            "--api-base",
            args.recipes_base_url,
            "--model",
            args.model,
            "--hardware",
            args.hardware,
        ]
    )

    return_code = 1
    started = False
    try:
        try:
            run(docker_cmd, timeout=120, capture_output=True)
            started = True
            result["stages"]["container_start"] = "PASS"
        except Exception as exc:
            result["stages"]["container_start"] = "FAIL"
            result["stages"]["conversion"] = "FAIL"
            result["error"] = f"docker_run: {exc}"
            return 1

        start = time.monotonic()
        ready = wait_ready(container, base_url, args.server_timeout)
        result["startup_seconds"] = round(time.monotonic() - start, 3)

        config_path = work / "config.yml"
        env_path = work / "env.sh"
        conversion_ok = config_path.is_file() and env_path.is_file()
        result["stages"]["conversion"] = "PASS" if conversion_ok else "FAIL"
        copy_candidates(work, result_dir)

        if not ready:
            result["stages"]["startup"] = "FAIL"
            if not conversion_ok:
                result["error"] = (
                    "serve_with_recipe.sh exited before generating config.yml/env.sh"
                )
            else:
                result["error"] = (
                    "vLLM did not become healthy before timeout or container exited"
                )
            return 1

        result["stages"]["startup"] = "PASS"
        result["stages"]["health"] = (
            "PASS"
            if docker_http_ok(container, f"{base_url}/health")
            else "FAIL"
        )
        result["stages"]["models_endpoint"] = (
            "PASS"
            if docker_http_ok(container, f"{base_url}/v1/models")
            else "FAIL"
        )

        try:
            served_model_name = get_served_model_name(
                container, base_url, args.model
            )
            result["served_model_name"] = served_model_name
            result["stages"]["served_model_resolution"] = "PASS"
        except Exception as exc:
            result["stages"]["served_model_resolution"] = "FAIL"
            result["error"] = f"served_model_resolution: {exc}"
            return 1

        try:
            version = run(
                ["docker", "exec", container, "vllm", "--version"],
                timeout=30,
                capture_output=True,
            ).stdout.strip()
            result["runtime"]["vllm_version"] = version
        except Exception as exc:
            result["runtime"]["vllm_version_error"] = str(exc)

        bench_dir = result_dir / "benchmark"
        bench_dir.mkdir(exist_ok=True)
        bench_log = result_dir / "benchmark.log"

        model_override = load_model_override(args.model)
        benchmark_override = model_override.get("benchmark", {})
        if not isinstance(benchmark_override, dict):
            raise ValueError(
                f"invalid benchmark override for {args.model!r}: expected an object"
            )

        backend = benchmark_override.get("backend", "vllm")
        endpoint = benchmark_override.get("endpoint", "/v1/completions")
        dataset_name = benchmark_override.get("dataset_name", "random")
        result["benchmark_profile"] = {
            "backend": backend,
            "endpoint": endpoint,
            "dataset_name": dataset_name,
        }

        python_packages = benchmark_override.get("python_packages", [])
        if not isinstance(python_packages, list) or not all(
            isinstance(package, str) and package.strip()
            for package in python_packages
        ):
            raise ValueError(
                f"invalid python_packages for {args.model!r}: "
                "expected a list of package names"
            )

        if python_packages:
            dependency_log = result_dir / "benchmark-dependencies.log"
            result["benchmark_profile"]["python_packages"] = python_packages
            try:
                with dependency_log.open("w") as log:
                    run(
                        [
                            "docker",
                            "exec",
                            container,
                            "uv",
                            "pip",
                            "install",
                            "--python",
                            "/opt/venv/bin/python",
                            *python_packages,
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=600,
                    )
                result["stages"]["benchmark_dependencies"] = "PASS"
            except Exception as exc:
                result["stages"]["benchmark_dependencies"] = "FAIL"
                result["error"] = f"benchmark_dependencies: {exc}"
                return 1
        else:
            result["stages"]["benchmark_dependencies"] = "SKIP"

        # Whisper's HF dataset loader may create lock files as container-root.
        # Isolate only the benchmark client's Hub and datasets caches in this
        # attempt's scratch workdir; keep the server's shared weight cache intact.
        benchmark_cache_env = []
        if args.model == "openai/whisper-large-v3":
            whisper_cache = work / "whisper-benchmark-hf-cache"
            whisper_cache.mkdir(parents=True, exist_ok=True)
            benchmark_cache_env = [
                "-e", "HF_HOME=/validation/whisper-benchmark-hf-cache",
                "-e", "HF_HUB_CACHE=/validation/whisper-benchmark-hf-cache/hub",
                "-e", "HF_DATASETS_CACHE=/validation/whisper-benchmark-hf-cache/datasets",
            ]

        bench_cmd = [
            "docker",
            "exec",
            # Trace only the serving engine and model runner, not the separate
            # benchmark client process or its tokenizer/helper kernels.
            *(["-e", "TRITON_RUNTIME_TRACE=0"]
              if os.environ.get("TRITON_TRACE_SITE_DIR") else []),
            *benchmark_cache_env,
            container,
            "vllm",
            "bench",
            "serve",
            "--backend",
            backend,
            "--base-url",
            base_url,
            "--model",
            args.model,
            "--served-model-name",
            served_model_name,
            "--endpoint",
            endpoint,
            "--dataset-name",
            dataset_name,
            "--num-prompts",
            str(args.num_prompts),
            "--request-rate",
            str(args.request_rate),
            "--max-concurrency",
            str(args.max_concurrency),
            "--percentile-metrics",
            "ttft,tpot",
            "--metric-percentiles",
            "99",
            "--save-result",
            "--save-detailed",
            "--result-dir",
            "/results/benchmark",
        ]

        if dataset_name == "random":
            bench_cmd.extend(
                [
                    "--random-input-len",
                    str(args.input_len),
                    "--random-output-len",
                    str(args.output_len),
                    "--ignore-eos",
                ]
            )
            if args.prefix_len > 0:
                bench_cmd.extend(["--random-prefix-len", str(args.prefix_len)])
        if args.num_warmups > 0:
            bench_cmd.extend(["--num-warmups", str(args.num_warmups)])

        dataset_path = benchmark_override.get("dataset_path")
        if dataset_path:
            bench_cmd.extend(["--dataset-path", str(dataset_path)])
            result["benchmark_profile"]["dataset_path"] = dataset_path

        hf_split = benchmark_override.get("hf_split")
        if hf_split:
            bench_cmd.extend(["--hf-split", str(hf_split)])
            result["benchmark_profile"]["hf_split"] = hf_split

        hf_subset = benchmark_override.get("hf_subset")
        if hf_subset:
            bench_cmd.extend(["--hf-subset", str(hf_subset)])
            result["benchmark_profile"]["hf_subset"] = hf_subset

        if benchmark_override.get("no_oversample"):
            bench_cmd.append("--no-oversample")
        if benchmark_override.get("no_stream"):
            bench_cmd.append("--no-stream")

        # ASR datasets use their own audio input duration rather than the text
        # random-input-len setting. Reuse output-len as the transcription token
        # cap so the quick validation remains bounded.
        if backend == "openai-audio":
            bench_cmd.extend(["--hf-output-len", str(args.output_len)])

        # The benchmark initializes its own tokenizer, independently of the
        # already-running server. Mirror the recipe-generated server setting
        # when the model requires Hugging Face custom code (for example,
        # microsoft/Phi-4-multimodal-instruct).
        trust_remote_code = any(
            line.startswith("trust-remote-code:")
            and line.split(":", 1)[1].strip().lower() == "true"
            for line in config_path.read_text().splitlines()
        )
        result["benchmark_trust_remote_code"] = trust_remote_code
        if trust_remote_code:
            bench_cmd.append("--trust-remote-code")

        # Optional, best-effort host perf sampling of native CPU execution.
        # Start AFTER readiness and BEFORE the inference probe; the collector
        # attaches to host PIDs visible under docker top for this container.
        native_perf = None
        perf_script = os.environ.get("TRITON_NATIVE_PERF_SCRIPT", "")
        if os.environ.get("TRITON_NATIVE_CPU_PERF") == "1" and perf_script:
            native_perf = subprocess.Popen([
                "python3", perf_script, "--container", container,
                "--output", str(result_dir / "native-perf"),
                "--seconds", str(max(60, args.bench_timeout)),
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)  # Allow perf to attach before inference traffic.
        try:
            with bench_log.open("w") as log:
                run(
                    bench_cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.bench_timeout,
                )
            bench_json = find_latest_json(bench_dir)
            if bench_json is None:
                raise RuntimeError("benchmark finished without a result JSON")
            raw = load_json(bench_json)
            result["benchmark_result_file"] = str(
                bench_json.relative_to(result_dir)
            )
            result["metrics"] = extract_metrics(raw)
            completed = result["metrics"].get("completed")
            # A zero-success benchmark may exit 0; never promote it as PASS.
            if isinstance(completed, bool) or not isinstance(completed, (int, float)) or completed <= 0:
                raise RuntimeError(
                    f"benchmark reported no successful requests "
                    f"(completed={completed!r})"
                )
            result["stages"]["benchmark"] = "PASS"
            if args.kvshrink_mode == "kvshrink":
                save_docker_logs(container, server_log_path)
                stats = kvshrink_log_stats(server_log_path)
                result["kvshrink"].update(stats)
                if not stats["connector_registered"]:
                    result["stages"]["kvshrink_connector"] = "FAIL"
                    raise RuntimeError("KVShrinkConnector did not register KV caches")
                result["stages"]["kvshrink_connector"] = "PASS"
                if args.prefix_len > 0 and stats["hit_requests"] == 0:
                    result["stages"]["kvshrink_hits"] = "FAIL"
                    raise RuntimeError("no request loaded KV blocks from KVShrink")
                result["stages"]["kvshrink_hits"] = (
                    "PASS" if stats["hit_requests"] else "SKIP"
                )
            result["status"] = "PASS"
            return_code = 0
        except Exception as exc:
            if result["stages"].get("benchmark") == "PASS":
                result["error"] = f"kvshrink: {exc}"
            else:
                result["stages"]["benchmark"] = "FAIL"
                result["error"] = f"benchmark: {exc}"
            return_code = 1
        finally:
            if native_perf is not None:
                if native_perf.poll() is None:
                    native_perf.send_signal(signal.SIGINT)
                try:
                    native_perf.wait(timeout=130)
                except subprocess.TimeoutExpired:
                    native_perf.kill()
                    native_perf.wait()
                # Collection failure must not mark a successful model probe
                # as failed; its JSON report records the profiler limitation.
        return return_code
    finally:
        if started:
            save_docker_logs(container, server_log_path)
            remove_container(container)
            if args.kvshrink_mode == "kvshrink":
                result["kvshrink"].update(kvshrink_log_stats(server_log_path))
        copy_candidates(work, result_dir)
        write_result(result_path, result)


if __name__ == "__main__":
    raise SystemExit(main())
