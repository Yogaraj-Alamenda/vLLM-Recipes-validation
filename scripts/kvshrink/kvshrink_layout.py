#!/usr/bin/env python3
"""Derive the KVShrink rank layout for a recipe-generated vLLM config.

Reads the parallel layout from the recipe ``config.yml`` and prints shell
``export`` lines for intel-accel-for-llm ``setvars.sh``:

- ``TP_SIZE`` / ``DP_SIZE`` / ``MODEL`` from the recipe config.
- ``VLLM_CPU_OMP_THREADS_BIND`` and ``KVSHRINK_CPU_OMP_THREADS_BIND``: one
  NUMA-local entry per global rank (``dp_rank * TP + tp_rank``). Each rank
  gets one logical CPU per physical core; the last ``KVSHRINK_IAXL_CORES``
  cores of every rank go to IAXL threads and the rest to vLLM inference, so
  the two never share a core.

Explicit ``VLLM_CPU_OMP_THREADS_BIND`` and ``KVSHRINK_CPU_OMP_THREADS_BIND``
values in the environment are kept as they are.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

import yaml

NODE_ROOT = Path("/sys/devices/system/node")
CPU_ROOT = Path("/sys/devices/system/cpu")


def expand_cpu_list(text: str) -> list[int]:
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def compress_cpu_list(cpus: list[int]) -> str:
    ranges: list[str] = []
    start = prev = None
    for cpu in sorted(cpus):
        if start is None:
            start = prev = cpu
        elif cpu == prev + 1:
            prev = cpu
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = cpu
    if start is not None:
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def numa_physical_cores() -> list[tuple[int, list[int]]]:
    """Return (node, [one allowed logical CPU per physical core]) per NUMA node."""
    allowed = os.sched_getaffinity(0)
    nodes = sorted(
        (int(p.name[4:]), p)
        for p in NODE_ROOT.glob("node[0-9]*")
        if (p / "cpulist").is_file()
    )
    result = []
    for node_id, path in nodes:
        seen: set[str] = set()
        cores: list[int] = []
        for cpu in expand_cpu_list((path / "cpulist").read_text()):
            if cpu not in allowed:
                continue
            siblings_file = CPU_ROOT / f"cpu{cpu}/topology/thread_siblings_list"
            siblings = (
                siblings_file.read_text().strip()
                if siblings_file.is_file()
                else str(cpu)
            )
            if siblings in seen:
                continue
            seen.add(siblings)
            cores.append(cpu)
        if cores:
            result.append((node_id, cores))
    return result


def split_ranks(
    nodes: list[tuple[int, list[int]]], ranks: int, iaxl_cores: int
) -> tuple[list[str], list[str], list[int]]:
    """Spread ranks over NUMA nodes in order; carve IAXL cores from each rank."""
    if not nodes:
        raise RuntimeError("no NUMA node with allowed CPUs was found")
    vllm_lists: list[str] = []
    iaxl_lists: list[str] = []
    rank_nodes: list[int] = []
    used_nodes = nodes[: min(ranks, len(nodes))]
    for index, (node_id, cores) in enumerate(used_nodes):
        node_ranks = ranks // len(used_nodes) + (index < ranks % len(used_nodes))
        per_rank, extra = divmod(len(cores), node_ranks)
        start = 0
        for local in range(node_ranks):
            count = per_rank + (local < extra)
            chunk = cores[start : start + count]
            start += count
            if len(chunk) <= iaxl_cores:
                raise RuntimeError(
                    f"NUMA node {node_id} gives a rank {len(chunk)} physical "
                    f"core(s); need more than KVSHRINK_IAXL_CORES={iaxl_cores}"
                )
            vllm_lists.append(compress_cpu_list(chunk[:-iaxl_cores]))
            iaxl_lists.append(compress_cpu_list(chunk[-iaxl_cores:]))
            rank_nodes.append(node_id)
    return vllm_lists, iaxl_lists, rank_nodes


def config_int(config: dict, *keys: str, default: int = 1) -> int:
    for key in keys:
        if key in config and config[key] is not None:
            return int(config[key])
    return default


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="Recipe-generated config.yml")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text()) or {}
    tp = config_int(config, "tensor-parallel-size", "tensor_parallel_size")
    dp = config_int(config, "data-parallel-size", "data_parallel_size")
    pp = config_int(config, "pipeline-parallel-size", "pipeline_parallel_size")
    if pp != 1:
        raise RuntimeError(
            f"pipeline-parallel-size={pp}: KVShrink rank binding supports TP x DP only"
        )
    model = config.get("model")
    if not model:
        raise RuntimeError(f"{args.config} does not set model")

    exports = {"TP_SIZE": str(tp), "DP_SIZE": str(dp), "MODEL": str(model)}

    vllm_bind = os.environ.get("VLLM_CPU_OMP_THREADS_BIND", "")
    iaxl_bind = os.environ.get("KVSHRINK_CPU_OMP_THREADS_BIND", "")
    if vllm_bind and vllm_bind != "auto" and iaxl_bind:
        layout = "environment"
    else:
        iaxl_cores = int(os.environ.get("KVSHRINK_IAXL_CORES", "2"))
        vllm_lists, iaxl_lists, rank_nodes = split_ranks(
            numa_physical_cores(), tp * dp, iaxl_cores
        )
        vllm_bind = "|".join(vllm_lists)
        iaxl_bind = "|".join(iaxl_lists)
        layout = "numa:" + ",".join(str(n) for n in rank_nodes)

    exports["VLLM_CPU_OMP_THREADS_BIND"] = vllm_bind
    exports["KVSHRINK_CPU_OMP_THREADS_BIND"] = iaxl_bind
    exports["KVSHRINK_RANK_LAYOUT"] = layout

    for name, value in exports.items():
        print(f"export {name}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # surface a single clear line in server.log
        print(f"kvshrink_layout: ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
