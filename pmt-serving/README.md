# ServeMR sweep — finding performance-promise violations in vLLM

`sweep.py` boots `vllm serve` under different optimization configs, runs
`vllm bench serve` (the hit-rate-extended version) on a shared-prefix workload,
and flags any **Performance Metamorphic Relation (PMR)** violation as a
candidate bug.

## Run (on the GPU server, inside the venv that has our vLLM branch)

```bash
source ~/koa-project/vllm-bench/.venv/bin/activate
export HF_HOME=/data1/koa/hf_cache
cd /data1/koa/pmt-serving           # copy sweep.py here

python sweep.py --model Qwen/Qwen2.5-7B-Instruct --reps 3 --gpu 0 --out runs/
```

Optional: pin GPU clocks first to cut performance noise (needs sudo):
`sudo nvidia-smi -i 0 -lgc 1500`

## What it checks (each violation = candidate bug)

| PMR | Promise |
|---|---|
| A1_cache_reduces_TTFT | prefix caching must lower TTFT on a shared-prefix workload |
| A1b_cache_engages | hit rate must be > 0 when caching is on (cf. #31920/#36493) |
| INT_chunked_prefill_erodes_cache | chunked prefill must not erode the cache's TTFT benefit (cf. #8223) |
| B1_mem_monotonic_hitrate | more KV space (gpu-mem 0.9 vs 0.5) must not reduce hit rate |
| A3_cudagraph_throughput | CUDA graphs must not reduce throughput vs enforce-eager |

## Output

- `runs/report.json` — per-config medians + PMR verdicts
- `runs/<config>.repN.bench.json` — raw bench results
- `runs/<config>.repN.serve.log` — server logs (for triage)

## Triage a flagged violation (before calling it a bug)

1. Re-run that config pair with more `--reps` to rule out **noise**.
2. Check it is within the optimization's **intended regime** (e.g., A1 only
   holds when prefix-len > 0). Not a bug if the regime doesn't apply.
3. Reproduce minimally, read `serve.log`, search the issue tracker.
4. If confirmed: file a vLLM issue/PR + record as a paper datapoint.
