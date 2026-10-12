#!/usr/bin/env python3
"""Report runtime Triton dispatches and optional sampled native CPU symbols."""
import argparse
import collections
import csv
import datetime as dt
import html
import json
import re
from pathlib import Path

LOG_V2 = re.compile(r'Using V2 Model Runner|V2ModelRunner|V2 model runner', re.I)
LOG_V1_ENGINE = re.compile(r'Initializing a V1 LLM engine', re.I)
FALLBACK = re.compile(r'fall(?:ing)? back to (?:the )?V1 (?:model )?runner', re.I)
IR = {'.ttir', '.ttgir', '.llir', '.ll', '.ptx', '.asm', '.s'}


def md(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def read_events(path):
    events = []
    if path.exists():
        for line in path.read_text(errors='replace').splitlines():
            try:
                events.append(json.loads(line))
            except (ValueError, TypeError):
                continue
    return events


def inspect(directory):
    result = json.loads((directory / 'result.json').read_text())
    log_file = directory / 'server.log'
    log = log_file.read_text(errors='replace') if log_file.exists() else ''
    events = read_events(directory / 'triton-launches.jsonl')
    jit = [e for e in events if e.get('event') == 'triton_jit_dispatch']
    compiled = [e for e in events if e.get('event') == 'triton_compiled_launch']
    # Lower-level launch dispatches take precedence; JIT events for the same
    # invocation are diagnostic only, not additive counts.
    lower_ids = {e.get('dispatch_id') for e in compiled if e.get('dispatch_id')}
    jit_only = [e for e in jit if e.get('dispatch_id') not in lower_ids]
    launches = sorted(compiled + jit_only, key=lambda e: (e.get('timestamp_ns', 0), e.get('pid', 0)))
    rows = []
    for i, e in enumerate(launches, 1):
        rows.append({'index': i, 'timestamp_ns': e.get('timestamp_ns'),
                     'pid': e.get('pid'), 'source': 'compiled' if e in compiled else 'jit-only',
                     'jit_name': e.get('jit_name') or '-', 'compiled_name': e.get('compiled_name') or '-',
                     'kernel_hash': e.get('kernel_hash') or '-', 'grid': e.get('grid'),
                     'backend': e.get('backend') or 'unknown', 'caller': e.get('caller') or '-'})
    with (directory / 'triton-launches.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [
            'index', 'timestamp_ns', 'pid', 'source', 'jit_name', 'compiled_name',
            'kernel_hash', 'grid', 'backend', 'caller'])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'grid': json.dumps(row['grid'])})
    count = collections.Counter((e['compiled_name'], e['kernel_hash'], e['jit_name'],
                                 json.dumps(e['grid']), e['source']) for e in rows)
    kernels = [{'compiled_name': key[0], 'kernel_hash': key[1], 'jit_name': key[2],
                'grid': json.loads(key[3]), 'source': key[4], 'launches': value}
               for key, value in count.most_common()]
    native_file = directory / 'native-perf' / 'native-cpu-symbols.json'
    native = json.loads(native_file.read_text()) if native_file.exists() else {
        'status': 'not_requested', 'symbols': []}
    native_top = native.get('symbols', [])[:10]
    dump = directory / 'triton'
    files = sorted(str(p.relative_to(directory)) for p in dump.rglob('*') if p.is_file()) if dump.exists() else []
    info = {
        'model': result.get('model', directory.name), 'probe': result.get('status', 'UNKNOWN'),
        'runner': 'V2 observed' if LOG_V2.search(log) else 'Unconfirmed',
        'v1_engine_banner': bool(LOG_V1_ENGINE.search(log)),
        'v1_fallback_observed': bool(FALLBACK.search(log)),
        'trace_status': 'Active' if any(e.get('event') == 'trace_ready' for e in events) else 'Unconfirmed',
        'trace_errors': [e for e in events if e.get('event') in ('trace_unavailable', 'trace_error')],
        'jit_events': len(jit), 'compiled_events': len(compiled),
        'jit_only_events': len(jit_only),
        'direct_compiled_events': sum(not e.get('dispatch_id') for e in compiled),
        'launches': len(rows),
        'unique_kernels': len({(k['compiled_name'], k['kernel_hash']) for k in kernels}),
        'kernels': kernels, 'launch_events': rows, 'native': native,
        'native_top': native_top, 'dump_files': len(files),
        'ir_files': sum(Path(n).suffix.lower() in IR for n in files),
        'files': files, 'error': result.get('error') or '-',
    }
    (directory / 'triton-inspection.json').write_text(json.dumps(info, indent=2) + '\n')
    return info


def main():
    p = argparse.ArgumentParser()
    for item in ('results', 'output', 'run-id', 'run-date', 'image'):
        p.add_argument('--' + item, required=True)
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    results = Path(a.results)
    models = [inspect(d) for d in sorted(results.iterdir()) if (d / 'result.json').exists()]
    summary = {
        'generated_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'run_id': a.run_id, 'run_date': a.run_date, 'image': a.image,
        'tested': len(models), 'probe_passed': sum(r['probe'] == 'PASS' for r in models),
        'triton_launches': sum(r['launches'] for r in models), 'models': models,
    }
    (output / 'triton-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    # Keep the top-level Actions summary focused on the two Triton paths,
    # native CPU symbols, and exact Triton kernel names. Detailed per-model
    # evidence remains available in JSON/CSV and the sections below.
    heads = ['Model', 'Probe', 'Runner', 'Triton JIT Dispatch',
             'Triton Direct Compiled', 'Native CPU Kernels (sampled)',
             'Exact Triton Kernels']
    lines = ['# Xeon 6 MRV2 Runtime Kernel Inspection', '',
             f"**Run:** `{md(a.run_id)}`  ", f"**Image:** `{md(a.image)}`  ",
             f"**Models:** {len(models)}; **Probe passed:** {summary['probe_passed']}; "
             f"**observed Triton dispatches:** {summary['triton_launches']}", '',
             '| ' + ' | '.join(heads) + ' |', '| ' + ' | '.join(['---'] * len(heads)) + ' |']
    html_rows = []
    for m in models:
        # Aggregate across grids/specializations so each exact kernel name
        # appears once in the summary column. Per-specialization data stays
        # in the detailed report and JSON/CSV artifacts.
        names = collections.Counter()
        for kernel in m['kernels']:
            names[kernel['compiled_name'] if kernel['compiled_name'] != '-'
                  else kernel['jit_name']] += kernel['launches']
        exact = ', '.join(f"`{md(name)}` ({count}×)"
                          for name, count in sorted(names.items())) or '—'
        nstat = m['native'].get('status', 'not_requested')
        native = ', '.join(f"`{md(n['symbol'])}` ({n['samples']})" for n in m['native_top'][:5]) or nstat
        values = [m['model'], m['probe'], m['runner'],
                  m['jit_events'], m['direct_compiled_events'], native, exact]
        lines.append('| ' + ' | '.join(md(v) for v in values) + ' |')
        html_rows.append('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in values) + '</tr>')
    lines += ['', '**Triton counters:** `Triton JIT Dispatch` counts recorded '
              '`JITFunction.run` calls. `Triton Direct Compiled` counts lower-level '
              'compiled dispatches *without a matching JIT dispatch ID*; these '
              'may be direct calls or correlation gaps. The two values do not '
              'represent native CPU execution and are not hardware-confirmed completions. '
              'The detailed artifacts retain all compiled dispatches and deduplicated counts.',
              '', '**Native CPU:** optional Linux `perf` sampling attaches to the serving '
              'container’s PIDs before inference. Symbols are sampled stack frames, **not** '
              'a complete kernel or launch inventory. Collection may be unavailable due to '
              'permissions or newly spawned worker processes.', '',
              '## Exact Triton runtime dispatches by model', '']
    for m in models:
        lines += [f"### {md(m['model'])}", '',
                  f"Triton launches: **{m['launches']}**; native profiling: "
                  f"**{md(m['native'].get('status', 'not_requested'))}**", '']
        if m['kernels']:
            lines += ['| Compiled kernel | JIT name | Hash | Grid | Source | Launches |',
                      '| --- | --- | --- | --- | --- | ---: |']
            for k in m['kernels']:
                lines.append('| ' + ' | '.join(md(v) for v in [k['compiled_name'], k['jit_name'],
                    k['kernel_hash'], json.dumps(k['grid']), k['source'], k['launches']]) + ' |')
        else:
            lines.append('No Triton dispatch observed by the installed Python hooks.')
        lines.append('')
        if m['native_top']:
            lines += ['**Top sampled native CPU symbols:**', '', '| Symbol | DSO | Frame occurrences |',
                      '| --- | --- | ---: |']
            for n in m['native_top']:
                lines.append('| ' + ' | '.join(md(v) for v in [n['symbol'], n['dso'], n['samples']]) + ' |')
        elif m['native'].get('error'):
            lines.append('Native sampler: ' + md(m['native']['error']))
        if m['trace_errors']:
            lines.append('Tracer warnings: ' + md('; '.join(str(e) for e in m['trace_errors'])))
        lines.append('')
    lines.append('Per-model `triton-launches.csv`, `triton-launches.jsonl`, '
                 '`triton-inspection.json`, `server.log`, and optional `native-perf/` '
                 'contain the underlying evidence. A V1 engine banner alone is not a runner fallback.')
    markdown = '\n'.join(lines) + '\n'
    (output / 'triton-summary.md').write_text(markdown)
    page = ('<!doctype html><html><head><meta charset="utf-8"><title>MRV2 Kernel Inspection</title>'
            '<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse;width:100%}'
            'td,th{border:1px solid #ccc;padding:.5rem;text-align:left;vertical-align:top;overflow-wrap:anywhere}'
            'th{background:#eee}</style></head><body><h1>Xeon 6 MRV2 Runtime Kernel Inspection</h1>'
            '<p>Triton Direct Compiled means unmatched lower-level compiled dispatches; '
            'native CPU symbols are sampled, not launch counts.</p>'
            '<table><thead><tr>' + ''.join(f'<th>{html.escape(h)}</th>' for h in heads) +
            '</tr></thead><tbody>' + ''.join(html_rows) + '</tbody></table>' +
            '<h2>Detailed Markdown report</h2><pre>' + html.escape(markdown) + '</pre></body></html>')
    (output / 'triton-summary.html').write_text(page)
    print(markdown)


if __name__ == '__main__':
    main()
