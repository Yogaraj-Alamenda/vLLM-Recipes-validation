#!/usr/bin/env python3
"""Run the KVShrink validation (recipe baseline vs recipe + KVShrink) per model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODES = ("baseline", "kvshrink")


def skip_reason(model: str) -> str | None:
    """The prefix-hit workload needs the default random text benchmark."""
    path = Path(__file__).resolve().parents[1] / "config/model_overrides.json"
    if not path.is_file():
        return None
    override = json.loads(path.read_text()).get("models", {}).get(model, {})
    benchmark = override.get("benchmark", {}) if isinstance(override, dict) else {}
    dataset = benchmark.get("dataset_name", "random")
    if dataset != "random":
        return f"benchmark override uses dataset {dataset!r}, not a prefix-hit workload"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", required=True)
    ap.add_argument("--vllm-dir", help="Optional checkout supplying tools/recipes")
    ap.add_argument("--image", required=True, help="KVShrink image used by both runs")
    ap.add_argument("--recipes-base-url", default="https://recipes.vllm.ai")
    ap.add_argument("--hf-home", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--hardware", default="xeon6")
    ap.add_argument("--input-len", type=int, default=8000, help="Total prompt tokens")
    ap.add_argument("--hit-rate", type=int, default=80, help="Shared-prefix percent")
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--num-warmups", type=int, default=5)
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--request-rate", default="inf")
    ap.add_argument("--server-timeout", type=int, default=3600)
    ap.add_argument("--bench-timeout", type=int, default=3600)
    ap.add_argument("--only", action="append", help="Test only exact model ID; repeatable")
    args = ap.parse_args()

    if not 0 < args.hit_rate < 100:
        print("--hit-rate must be between 1 and 99", file=sys.stderr)
        return 2
    prefix_len = args.input_len * args.hit_rate // 100
    random_len = args.input_len - prefix_len

    models = json.loads(Path(args.discovery).read_text())["models"]
    if args.only:
        wanted = set(args.only)
        models = [m for m in models if m["hf_id"] in wanted]
        missing = wanted - {m["hf_id"] for m in models}
        if missing:
            print(
                "Requested model(s) not found in Xeon6 discovery: "
                + ", ".join(sorted(missing)),
                file=sys.stderr,
            )
            return 2

    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_one_model.py")
    failures: list[str] = []
    skipped: list[dict] = []
    for i, item in enumerate(models, start=1):
        model = item["hf_id"]
        print(f"\n===== [{i}/{len(models)}] {model} =====", flush=True)
        reason = skip_reason(model)
        if reason:
            print(f"SKIP: {model}: {reason}")
            skipped.append({"model": model, "reason": reason})
            continue

        model_failed = False
        for mode in MODES:
            print(f"--- {mode} ---", flush=True)
            cmd = [
                sys.executable, str(runner),
                "--model", model,
                "--recipe-url", item["recipe_url"],
                "--image", args.image,
                "--recipes-base-url", args.recipes_base_url,
                "--hf-home", args.hf_home,
                "--work-root", str(Path(args.work_root) / mode),
                "--result-root", str(result_root / mode),
                "--hardware", args.hardware,
                "--input-len", str(random_len),
                "--prefix-len", str(prefix_len),
                "--output-len", str(args.output_len),
                "--num-prompts", str(args.num_prompts),
                "--num-warmups", str(args.num_warmups),
                "--max-concurrency", str(args.max_concurrency),
                "--request-rate", str(args.request_rate),
                "--server-timeout", str(args.server_timeout),
                "--bench-timeout", str(args.bench_timeout),
                "--kvshrink-mode", mode,
            ]
            if args.vllm_dir:
                cmd.extend(["--vllm-dir", args.vllm_dir])
            rc = subprocess.run(cmd, env=os.environ.copy()).returncode
            print(f"{'FAIL' if rc else 'PASS'}: {model} [{mode}]", flush=True)
            model_failed |= rc != 0
        if model_failed:
            failures.append(model)

    summary = {
        "tested": len(models) - len(skipped),
        "passed": len(models) - len(skipped) - len(failures),
        "failed": len(failures),
        "failed_models": failures,
        "skipped": skipped,
        "workload": {
            "input_tokens": args.input_len,
            "hit_rate_percent": args.hit_rate,
            "prefix_tokens": prefix_len,
            "random_tokens": random_len,
            "output_tokens": args.output_len,
            "num_prompts": args.num_prompts,
            "num_warmups": args.num_warmups,
            "max_concurrency": args.max_concurrency,
            "request_rate": args.request_rate,
        },
    }
    (result_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
