#!/usr/bin/env python3
"""Build the KVShrink validation JSON + HTML summary from per-mode result.json files."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
from pathlib import Path

MODES = ("baseline", "kvshrink")
METRICS = (
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "output_throughput",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def ratio(num, den):
    if isinstance(num, (int, float)) and isinstance(den, (int, float)) and den:
        return round(num / den, 3)
    return None


def pct_change(new, old):
    if isinstance(new, (int, float)) and isinstance(old, (int, float)) and old:
        return round((new - old) / old * 100, 1)
    return None


def tensor_parallel(result_dir: Path):
    config = result_dir / "config.yml"
    if not config.is_file():
        return None
    for line in config.read_text().splitlines():
        if line.startswith("tensor-parallel-size:"):
            return line.split(":", 1)[1].strip()
    return "1"


def model_row(model: str, safe: str, results: Path) -> dict:
    runs = {mode: load(results / mode / safe / "result.json") for mode in MODES}
    base = runs["baseline"].get("metrics", {})
    kvs = runs["kvshrink"].get("metrics", {})
    statuses = {mode: runs[mode].get("status", "MISSING") for mode in MODES}
    return {
        "model": model,
        "status": "PASS" if all(s == "PASS" for s in statuses.values()) else "FAIL",
        "mode_status": statuses,
        "errors": {mode: runs[mode].get("error") for mode in MODES if runs[mode].get("error")},
        "tensor_parallel_size": tensor_parallel(results / "kvshrink" / safe)
        or tensor_parallel(results / "baseline" / safe),
        "baseline": {k: base.get(k) for k in METRICS},
        "kvshrink": {k: kvs.get(k) for k in METRICS},
        "kvshrink_stats": runs["kvshrink"].get("kvshrink", {}),
        "comparison": {
            "mean_ttft_speedup": ratio(base.get("mean_ttft_ms"), kvs.get("mean_ttft_ms")),
            "p99_ttft_speedup": ratio(base.get("p99_ttft_ms"), kvs.get("p99_ttft_ms")),
            "mean_tpot_change_pct": pct_change(kvs.get("mean_tpot_ms"), base.get("mean_tpot_ms")),
            "p99_tpot_change_pct": pct_change(kvs.get("p99_tpot_ms"), base.get("p99_tpot_ms")),
            "output_throughput_ratio": ratio(
                kvs.get("output_throughput"), base.get("output_throughput")
            ),
        },
    }


def fmt(value, suffix=""):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value}{suffix}"


def render_html(summary: dict) -> str:
    w = summary.get("workload", {})
    image = summary.get("kvshrink_image", {})
    rows = []
    for r in summary["models"]:
        b, k, c, s = r["baseline"], r["kvshrink"], r["comparison"], r["kvshrink_stats"]
        cls = "pass" if r["status"] == "PASS" else "fail"
        errors = "; ".join(f"{a}: {e}" for a, e in r["errors"].items())
        rows.append(
            "<tr class='{}'>".format(cls)
            + "".join(
                f"<td>{html.escape(str(v))}</td>"
                for v in (
                    r["model"],
                    f"{r['mode_status']['baseline']} / {r['mode_status']['kvshrink']}",
                    r["tensor_parallel_size"] or "-",
                    fmt(b["mean_ttft_ms"]),
                    fmt(k["mean_ttft_ms"]),
                    fmt(c["mean_ttft_speedup"], "x"),
                    fmt(b["p99_ttft_ms"]),
                    fmt(k["p99_ttft_ms"]),
                    fmt(c["p99_ttft_speedup"], "x"),
                    fmt(b["mean_tpot_ms"]),
                    fmt(k["mean_tpot_ms"]),
                    fmt(c["mean_tpot_change_pct"], "%"),
                    fmt(c["output_throughput_ratio"], "x"),
                    f"{s.get('hit_requests', '-')}/{s.get('lookups', '-')}",
                    fmt(s.get("external_tokens")),
                    errors or "-",
                )
            )
            + "</tr>"
        )
    skipped = "".join(
        f"<li>{html.escape(x['model'])}: {html.escape(x['reason'])}</li>"
        for x in summary.get("skipped", [])
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>KVShrink validation - Xeon 6 recipes</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px}}
table{{border-collapse:collapse;font-size:13px}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}
td:first-child,th:first-child,td:last-child{{text-align:left}}
tr.pass{{background:#eef9ee}}tr.fail{{background:#fdecec}}
</style></head><body>
<h1>KVShrink validation on Xeon 6 vLLM recipes</h1>
<p>Run <code>{html.escape(summary['run_id'])}</code> generated {html.escape(summary['generated_at'])}.
Tested {summary['tested']}, passed {summary['passed']}, failed {summary['failed']}.</p>
<p>Workload: {w.get('input_tokens')} input tokens ({w.get('hit_rate_percent')}% shared prefix =
{w.get('prefix_tokens')} tokens), {w.get('output_tokens')} output tokens, {w.get('num_prompts')} prompts,
{w.get('num_warmups')} warmups, max concurrency {w.get('max_concurrency')}.
Prefix caching: {html.escape(str(summary.get('prefix_caching')))} in both runs.</p>
<p>Image: <code>{html.escape(str(image.get('image')))}</code>
(KVShrink {html.escape(str(image.get('source')))} {html.escape(str(image.get('ref') or ''))}
{html.escape(str(image.get('commit') or ''))}{' dirty' if image.get('dirty') else ''};
base <code>{html.escape(str(image.get('base_image')))}</code>)</p>
<table><tr><th>Model</th><th>Status (base / KVShrink)</th><th>TP</th>
<th>Mean TTFT base ms</th><th>Mean TTFT KVShrink ms</th><th>Mean TTFT speedup</th>
<th>P99 TTFT base ms</th><th>P99 TTFT KVShrink ms</th><th>P99 TTFT speedup</th>
<th>Mean TPOT base ms</th><th>Mean TPOT KVShrink ms</th><th>TPOT change</th>
<th>Output tok/s ratio</th><th>KVShrink hit requests / lookups</th><th>KVShrink tokens</th><th>Error</th></tr>
{''.join(rows)}</table>
{f'<h2>Skipped</h2><ul>{skipped}</ul>' if skipped else ''}
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Directory with baseline/ and kvshrink/")
    ap.add_argument("--image-metadata", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-date", required=True)
    ap.add_argument("--prefix-caching", default="off")
    args = ap.parse_args()

    results = Path(args.results)
    run_summary = load(results / "run_summary.json")
    safes = sorted(
        {p.parent.name for mode in MODES for p in (results / mode).glob("*/result.json")}
    )
    models = []
    for safe in safes:
        model = next(
            (
                load(results / mode / safe / "result.json").get("model")
                for mode in MODES
                if (results / mode / safe / "result.json").is_file()
            ),
            safe.replace("__", "/", 1),
        )
        models.append(model_row(model, safe, results))

    passed = sum(1 for m in models if m["status"] == "PASS")
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": args.run_id,
        "run_date": args.run_date,
        "tested": len(models),
        "passed": passed,
        "failed": len(models) - passed,
        "skipped": run_summary.get("skipped", []),
        "workload": run_summary.get("workload", {}),
        "prefix_caching": args.prefix_caching,
        "kvshrink_image": load(Path(args.image_metadata)),
        "models": models,
    }
    (results / "kvshrink-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (results / "kvshrink-summary.html").write_text(render_html(summary))
    print(f"Wrote {results / 'kvshrink-summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
