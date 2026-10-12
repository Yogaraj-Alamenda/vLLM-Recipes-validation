#!/usr/bin/env python3
"""Sample native CPU execution symbols in a vLLM container using host perf.

Best-effort: CAP_PERFMON/perf_event_paranoid, host perf and frame pointers
may restrict the results. Never claim that sampled symbols are exhaustive.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def collect_pids(container):
    proc = subprocess.run(['docker', 'top', container, '-eo', 'pid'],
                          capture_output=True, text=True, check=True, timeout=15)
    return [x.strip() for x in proc.stdout.splitlines()[1:] if x.strip().isdigit()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--container', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seconds', type=int, default=1800)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = {'status': 'unavailable', 'symbols': [], 'sampling': 'perf cpu-clock sampled symbols, not complete kernel inventory'}
    report = out / 'native-cpu-symbols.json'
    try:
        if not shutil.which('perf'):
            raise RuntimeError('host perf executable unavailable')
        pids = collect_pids(a.container)
        if not pids:
            raise RuntimeError('no host PIDs found for container')
        meta['pids'] = pids
        perf_data = out / 'native-perf.data'
        cmd = ['perf', 'record', '-q', '-e', 'cpu-clock:u', '-F', '49', '-g',
               '--call-graph', 'fp', '-p', ','.join(pids), '-o', str(perf_data), '--',
               'sleep', str(a.seconds)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                start_new_session=True)
        # The benchmark runner sends SIGINT to stop collection after inference.
        def stop(_sig, _frame):
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        _, err = proc.communicate()
        if not perf_data.exists() or not perf_data.stat().st_size:
            raise RuntimeError('perf recording missing; ' + err.decode(errors='replace')[-500:])
        sample = subprocess.run(['perf', 'script', '-i', str(perf_data), '-F', 'sym,dso'],
                                capture_output=True, text=True, timeout=120)
        if sample.returncode:
            raise RuntimeError('perf script: ' + sample.stderr[-500:])
        symbols = collections.Counter()
        for line in sample.stdout.splitlines():
            parts = line.strip().split(' (', 1)
            if len(parts) != 2:
                continue
            symbol = parts[0].strip()
            dso = parts[1].rstrip(')').strip()
            if not symbol or symbol in ('[unknown]', '[kernel.kallsyms]'):
                continue
            symbols[(symbol, dso)] += 1
        meta['status'] = 'sampled' if symbols else 'no_symbols'
        meta['symbols'] = [{'symbol': k[0], 'dso': k[1], 'samples': n}
                           for k, n in symbols.most_common(100)]
        meta['sampling_note'] = 'Stacks may contain multiple frames per sample; counts are frame occurrences, not launches.'
        if err:
            meta['perf_stderr_tail'] = err.decode(errors='replace')[-500:]
    except Exception as exc:
        meta['error'] = str(exc)
    finally:
        report.write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
