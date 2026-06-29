#!/usr/bin/env python3
"""Optimization-interaction sweep: find performance-promise violations in vLLM.

For each configuration it boots `vllm serve`, runs `vllm bench serve` on a
shared-prefix workload (reusing the hit-rate metric we added), tears the server
down, and records TTFT / throughput / prefix-cache-hit-rate. It then checks a
set of Performance Metamorphic Relations (PMRs) across configs and flags any
"enabling an optimization made things worse" cell as a candidate bug.

Run on the GPU server inside the activated venv:

    python sweep.py --model Qwen/Qwen2.5-7B-Instruct --reps 3 --out runs/

Each flagged PMR violation is a *candidate* bug -> triage (rule out noise /
legitimate trade-off) -> reproduce -> PR + paper datapoint.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# --- Configurations under test ---------------------------------------------
# Each maps a name to extra `vllm serve` flags. Attributes drive the PMR logic.
# prefix caching + chunked prefill are ON by default in v1; we toggle explicitly.
CONFIGS: dict[str, dict] = {
    "base": {  # cache on, chunk on, cudagraph on, mem 0.9  -> the reference
        "args": ["--enable-prefix-caching", "--enable-chunked-prefill",
                 "--gpu-memory-utilization", "0.9"],
    },
    "no_cache": {  # cache OFF, chunk on
        "args": ["--no-enable-prefix-caching", "--enable-chunked-prefill",
                 "--gpu-memory-utilization", "0.9"],
    },
    "no_chunk": {  # cache on, chunk OFF
        "args": ["--enable-prefix-caching", "--no-enable-chunked-prefill",
                 "--gpu-memory-utilization", "0.9"],
    },
    "no_cache_no_chunk": {  # cache OFF, chunk OFF
        "args": ["--no-enable-prefix-caching", "--no-enable-chunked-prefill",
                 "--gpu-memory-utilization", "0.9"],
    },
    "eager": {  # cudagraph OFF (enforce eager)
        "args": ["--enable-prefix-caching", "--enable-chunked-prefill",
                 "--enforce-eager", "--gpu-memory-utilization", "0.9"],
    },
    "lowmem": {  # smaller KV cache space
        "args": ["--enable-prefix-caching", "--enable-chunked-prefill",
                 "--gpu-memory-utilization", "0.5"],
    },
}

METRIC_KEYS = ("median_ttft_ms", "mean_ttft_ms", "output_throughput",
               "request_throughput", "total_token_throughput",
               "prefix_cache_hit_rate", "prefix_cache_queries",
               "prefix_cache_hits")


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_health(port: int, timeout: float, proc: subprocess.Popen) -> bool:
    """Poll /health until ready; fail fast if the server process dies."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False  # process exited before becoming healthy
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def run_config(name: str, cfg: dict, a: argparse.Namespace, rep: int) -> dict | None:
    """Boot serve for one config, run one bench, return metrics dict or None."""
    outdir = Path(a.out)
    log_path = outdir / f"{name}.rep{rep}.serve.log"
    result_name = f"{name}.rep{rep}.bench.json"
    result_path = outdir / result_name

    env = dict(os.environ)
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # server CUDA 11.8 -> avoid JIT crash
    env["CUDA_VISIBLE_DEVICES"] = a.gpu

    serve_cmd = [a.vllm_bin, "serve", a.model, "--port", str(a.port),
                 "--max-model-len", str(a.max_model_len), *cfg["args"]]

    print(f"\n=== [{name}] rep {rep}: starting serve ===", flush=True)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(serve_cmd, env=env, stdout=logf,
                                stderr=subprocess.STDOUT, start_new_session=True)
    try:
        if not wait_health(a.port, a.startup_timeout, proc):
            print(f"!! [{name}] server failed to start (see {log_path})", flush=True)
            return None

        bench_cmd = [a.vllm_bin, "bench", "serve", "--model", a.model,
                     "--port", str(a.port), "--dataset-name", "random",
                     "--num-prompts", str(a.num_prompts),
                     "--random-input-len", str(a.input_len),
                     "--random-output-len", str(a.output_len),
                     "--random-prefix-len", str(a.prefix_len),
                     "--max-concurrency", str(a.concurrency),
                     "--save-result", "--result-dir", str(outdir),
                     "--result-filename", result_name]
        print(f"--- [{name}] rep {rep}: running bench ---", flush=True)
        r = subprocess.run(bench_cmd, env=env)
        if r.returncode != 0 or not result_path.exists():
            print(f"!! [{name}] bench failed (rc={r.returncode})", flush=True)
            return None
        with open(result_path) as f:
            data = json.load(f)
        return {k: data.get(k) for k in METRIC_KEYS}
    finally:
        # Tear down the whole process group and wait for the port to free.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=30)
        except Exception:
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        for _ in range(60):
            if port_is_free(a.port):
                break
            time.sleep(1)


def median_of(reps: list[dict], key: str) -> float | None:
    vals = [r[key] for r in reps if r and r.get(key) is not None]
    return statistics.median(vals) if vals else None


def check_pmrs(agg: dict[str, dict], tol: float) -> list[dict]:
    """Evaluate Performance Metamorphic Relations. Violations are candidate bugs."""
    findings = []

    def m(cfg, key):
        return agg.get(cfg, {}).get(key)

    def add(pmr, ok, detail):
        findings.append({"pmr": pmr, "status": "OK" if ok else "VIOLATION",
                         "detail": detail})

    # A1: prefix caching must reduce TTFT on a shared-prefix workload.
    t_base, t_nocache = m("base", "median_ttft_ms"), m("no_cache", "median_ttft_ms")
    if t_base and t_nocache:
        ok = t_base <= t_nocache * (1 + tol)
        add("A1_cache_reduces_TTFT", ok,
            f"TTFT base(cache on)={t_base:.1f}ms vs no_cache={t_nocache:.1f}ms")

    # A1b: caching must actually engage (hit rate > 0) on a shared-prefix workload.
    hr = m("base", "prefix_cache_hit_rate")
    if hr is not None:
        add("A1b_cache_engages", hr > 0, f"base hit_rate={hr:.2f}% (expected >0)")

    # INT: does chunked prefill erode the TTFT benefit of prefix caching? (#8223)
    ben_chunk = (t_nocache - t_base) if (t_base and t_nocache) else None
    t_nochunk, t_neither = m("no_chunk", "median_ttft_ms"), m("no_cache_no_chunk", "median_ttft_ms")
    ben_nochunk = (t_neither - t_nochunk) if (t_nochunk and t_neither) else None
    if ben_chunk is not None and ben_nochunk is not None:
        # cache benefit should not be much smaller WITH chunked prefill than without
        ok = ben_chunk >= ben_nochunk * (1 - 0.25)
        add("INT_chunked_prefill_erodes_cache", ok,
            f"cache TTFT benefit: with chunked={ben_chunk:.1f}ms, "
            f"without chunked={ben_nochunk:.1f}ms")

    # B1: more KV cache space must not reduce hit rate (monotonicity).
    hr_hi, hr_lo = m("base", "prefix_cache_hit_rate"), m("lowmem", "prefix_cache_hit_rate")
    if hr_hi is not None and hr_lo is not None:
        ok = hr_hi >= hr_lo * (1 - tol)
        add("B1_mem_monotonic_hitrate", ok,
            f"hit_rate mem0.9={hr_hi:.2f}% vs mem0.5={hr_lo:.2f}%")

    # A3: CUDA graphs must not reduce throughput vs enforce-eager.
    tp_base, tp_eager = m("base", "output_throughput"), m("eager", "output_throughput")
    if tp_base and tp_eager:
        ok = tp_base >= tp_eager * (1 - tol)
        add("A3_cudagraph_throughput", ok,
            f"throughput cudagraph={tp_base:.1f} vs eager={tp_eager:.1f} tok/s")

    return findings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--out", default="runs")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--vllm-bin", default="vllm")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=64)
    p.add_argument("--prefix-len", type=int, default=512, help="shared prefix tokens")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--tol", type=float, default=0.05, help="relative tolerance")
    p.add_argument("--configs", default="", help="comma list to restrict configs")
    a = p.parse_args()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    names = [c for c in (a.configs.split(",") if a.configs else CONFIGS) if c in CONFIGS]

    raw: dict[str, list[dict]] = {}
    for name in names:
        raw[name] = []
        for rep in range(a.reps):
            res = run_config(name, CONFIGS[name], a, rep)
            if res:
                raw[name].append(res)

    agg = {name: {k: median_of(reps, k) for k in METRIC_KEYS}
           for name, reps in raw.items() if reps}

    findings = check_pmrs(agg, a.tol)

    report = {"model": a.model, "reps": a.reps,
              "workload": {"input_len": a.input_len, "output_len": a.output_len,
                           "prefix_len": a.prefix_len, "num_prompts": a.num_prompts,
                           "concurrency": a.concurrency},
              "aggregated_medians": agg, "pmr_findings": findings}
    (Path(a.out) / "report.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print("PER-CONFIG MEDIANS")
    for name, mtr in agg.items():
        ttft = mtr.get("median_ttft_ms")
        tp = mtr.get("output_throughput")
        hrr = mtr.get("prefix_cache_hit_rate")
        print(f"  {name:20s} TTFT={ttft!s:>8} tok/s={tp!s:>8} hit%={hrr!s:>7}")
    print("\nPMR FINDINGS")
    for f in findings:
        mark = "  OK " if f["status"] == "OK" else ">>BUG"
        print(f"  [{mark}] {f['pmr']}: {f['detail']}")
    viols = [f for f in findings if f["status"] == "VIOLATION"]
    print(f"\n{len(viols)} candidate bug(s). Full report: {Path(a.out)/'report.json'}")


if __name__ == "__main__":
    main()
