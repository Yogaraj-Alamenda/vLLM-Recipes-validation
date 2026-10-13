#!/usr/bin/env python3
"""Promote PASS bundles and generate JSON/HTML validation reports."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--validated", required=True)
    ap.add_argument("--history", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-date", required=True)
    ap.add_argument("--recipe-tools-metadata", required=True)
    args = ap.parse_args()

    recipe_tools = json.loads(Path(args.recipe_tools_metadata).read_text())
    results_root = Path(args.results)
    validated_root = Path(args.validated)
    history_root = Path(args.history) / args.run_date / args.run_id
    history_models = history_root / "models"
    history_models.mkdir(parents=True, exist_ok=True)
    validated_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for result_file in sorted(results_root.glob("*/result.json")):
        data = json.loads(result_file.read_text())
        model = data["model"]
        safe = model.replace("/", "__")
        src_dir = result_file.parent
        hist_dir = history_models / safe
        if hist_dir.exists():
            shutil.rmtree(hist_dir)
        shutil.copytree(src_dir, hist_dir)

        data["recipe_tools"] = recipe_tools
        data["run_id"] = args.run_id
        data["run_date"] = args.run_date
        (hist_dir / "result.json").write_text(json.dumps(data, indent=2) + "\n")

        if data.get("status") == "PASS":
            # Preserve namespace/model hierarchy for user-friendly retrieval.
            dest = validated_root.joinpath(*model.split("/"))
            tmp = dest.with_name(dest.name + ".tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            for name in (
                "config.yml",
                "env.sh",
                "recipe.json",
                "result.json",
                "system-info.txt",
            ):
                src = hist_dir / name
                if src.exists():
                    shutil.copy2(src, tmp / name)
            validation = {
                "model": model,
                "status": "validated",
                "validated_at": data.get("finished_at"),
                "run_id": args.run_id,
                "run_date": args.run_date,
                "hardware": data.get("hardware"),
                "recipe_tools": recipe_tools,
                "recipe_url": data.get("recipe_url"),
                "workload": data.get("workload"),
                "stages": data.get("stages"),
                "metrics": data.get("metrics", {}),
                "runtime": data.get("runtime", {}),
            }
            (tmp / "validation.json").write_text(
                json.dumps(validation, indent=2) + "\n"
            )
            (tmp / "README.md").write_text(
                f"# {model} — Xeon 6 validated configuration\n\n"
                f"Last validated: {validation['validated_at']}\n\n"
                f"Validation run: `{args.run_date}/{args.run_id}`\n\n"
                f"Recipe tools source: `{recipe_tools['source']}`\n\n"
                f"Recipe tools repository: `{recipe_tools.get('repository') or 'image-bundled'}`\n\n"
                f"Recipe tools ref / commit: `{recipe_tools.get('ref') or '-'} / {recipe_tools.get('commit') or '-'}`\n\n"
                f"Validated runtime image: "
                f"`{data.get('runtime', {}).get('image', 'unknown')}`\n\n"
                "The weekly validation generated and served this configuration via "
                "`tools/recipes/serve_with_recipe.sh`.\n"
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                shutil.rmtree(dest)
            tmp.rename(dest)

        rows.append(data)

    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    summary = {
        "generated_at": generated_at,
        "run_id": args.run_id,
        "run_date": args.run_date,
        "history_path": f"history/{args.run_date}/{args.run_id}",
        "recipe_tools": recipe_tools,
        "tested": len(rows),
        "passed": sum(r.get("status") == "PASS" for r in rows),
        "failed": sum(r.get("status") != "PASS" for r in rows),
        "models": rows,
    }
    latest_json = results_root / "weekly-summary.json"
    latest_json.write_text(json.dumps(summary, indent=2) + "\n")

    trs = []
    for r in rows:
        status = r.get("status", "FAIL")
        m = r.get("metrics", {})
        stages = r.get("stages", {})
        error = html.escape(str(r.get("error", ""))) or "-"
        trs.append(
            "<tr>"
            f"<td><code>{html.escape(r['model'])}</code></td>"
            f"<td class='{status.lower()}'>{html.escape(status)}</td>"
            f"<td>{html.escape(str(stages.get('conversion','-')))}</td>"
            f"<td>{html.escape(str(stages.get('startup','-')))}</td>"
            f"<td>{html.escape(str(stages.get('benchmark','-')))}</td>"
            f"<td>{html.escape(str(m.get('p99_ttft_ms','-')))}</td>"
            f"<td>{html.escape(str(m.get('p99_tpot_ms','-')))}</td>"
            f"<td>{html.escape(str(m.get('output_throughput','-')))}</td>"
            f"<td class='error'>{error}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Xeon 6 Recipe Validation — {html.escape(args.run_id)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;color:#1f2328}}
.summary{{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}}
.card{{border:1px solid #d0d7de;border-radius:8px;padding:.7rem 1rem;min-width:8rem}}
table{{border-collapse:collapse;width:100%;font-size:.92rem}}
th,td{{border:1px solid #d0d7de;padding:.5rem;text-align:left;vertical-align:top}}
th{{background:#f6f8fa;position:sticky;top:0}}
.pass{{color:#1a7f37;font-weight:700}} .fail{{color:#cf222e;font-weight:700}}
.error{{max-width:36rem;word-break:break-word}} code{{font-family:ui-monospace,monospace}}
</style></head><body>
<h1>Xeon 6 Recipe Validation</h1>
<p>Run: <code>{html.escape(args.run_id)}</code> &nbsp; Date: <code>{html.escape(args.run_date)}</code><br>
Recipe tools source: <code>{html.escape(recipe_tools["source"])}</code><br>
Repository: <code>{html.escape(recipe_tools.get("repository") or "image-bundled")}</code><br>
Ref / commit: <code>{html.escape(recipe_tools.get("ref") or "-")} / {html.escape(recipe_tools.get("commit") or "-")}</code><br>
Generated: <code>{html.escape(generated_at)}</code></p>
<div class='summary'>
<div class='card'><strong>Tested</strong><br>{len(rows)}</div>
<div class='card'><strong>Passed</strong><br>{summary['passed']}</div>
<div class='card'><strong>Failed</strong><br>{summary['failed']}</div>
</div>
<p>Quick performance workload: 128 input / 128 output, concurrency 1 by default; each result records the exact workload.</p>
<table><thead><tr><th>Model</th><th>Status</th><th>Convert</th><th>Startup</th><th>Benchmark</th><th>P99 TTFT ms</th><th>P99 TPOT ms</th><th>Output tok/s</th><th>Error</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table>
</body></html>"""
    latest_html = results_root / "weekly-summary.html"
    latest_html.write_text(page)

    # Keep report copies with this exact run. Per-model evidence is already
    # copied under history/<date>/<run-id>/models/ above.
    shutil.copy2(latest_json, history_root / "weekly-summary.json")
    shutil.copy2(latest_html, history_root / "weekly-summary.html")

    print(
        f"Published {summary['passed']} validated bundles; "
        f"{summary['failed']} failures; history={summary['history_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
