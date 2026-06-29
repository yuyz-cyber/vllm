# ServeMR — Performance Metamorphic Testing for LLM Serving Engines

> Working title. A methodology + tool to automatically find **performance bugs** in
> LLM inference engines (vLLM first) by checking that optimizations keep their
> implicit **performance promises**, expressed as Performance Metamorphic Relations.

## 1. Problem & Motivation

Mature infrastructure (databases, compilers, OSes) is backed by rigorous
performance-regression / correctness testing *disciplines*. AI infra (LLM serving
engines) is being built at high speed, dominated by performance optimizations
(prefix caching, chunked prefill, continuous batching, speculative decoding,
CUDA graphs, quantization), but its testing is largely crash-based and ad-hoc.

Evidence (Gao et al., *A First Look at Bugs in LLM Inference Engines*, 2506.09713):

- "Abnormal Performance" (S4) bugs have the **highest mean fix time (58.2 days)**;
  resource bugs 79.5 days. Rare to find, very costly.
- **>35% of bugs are non-crash**; "crash-focused testing misses performance
  degradation." KV-cache mgmt (RE.2, 11 cases) and concurrency (RB.3, 39 cases)
  are recurring root causes.
- Real instance of our target class: **DeepSpeed #2488 — the "optimized" model was
  *slower* than vanilla** (8566→8811ms). That is literally a performance-promise
  violation.

User-reported vLLM performance bugs that motivate concrete relations:

- #31920, #36493 — prefix-cache hit rate stuck at ~0 on identical/multi-turn prompts.
- #24394 — "dirty cache" hurts hit rate.
- #8223 — chunked prefill + prefix caching together *worsen* TTFT.

## 2. Core Idea

Every serving optimization carries an **implicit performance promise** ("turn this
on / give it more resources and metric M should improve, or at least not get
worse — in the regime it targets"). These promises are **Performance Metamorphic
Relations (PMRs)**. An engine that violates its own promise has a **performance
bug**, detectable **without a ground-truth oracle** — we only compare two runs.

Because we compare **performance metrics** (TTFT, throughput, hit rate, memory),
not token outputs, we **sidestep the batch-invariance nondeterminism** that
confounds output-equivalence testing (Thinking Machines, 2025).

## 3. Novelty & Positioning

Reference paradigm = PagedAttention / DLSmith: take an **established technique**
and apply it to a **new, important, under-tested domain** with domain-specific
adaptation.

| Prior work | What it is | Why we differ |
|---|---|---|
| Troya et al., *Performance Metamorphic Testing* (IST 2018) | General PMR concept + stats + seeded-fault validation | We **reuse** it as our scaffold; novelty is the **serving-specific PMR catalog + regime predicates + LLM-serving challenges**, not the general method |
| DLSmith (ISSTA 2023) | Metamorphic testing of **Datalog engines** | Same template ("MT of [system class]") → precedent that this paper shape is accepted |
| GRIEF (2605.11202) | **Stateful fuzzing for security/availability** vulns (isolation, DoS, liveness) | We target **performance correctness** of optimizations, not security; complementary (their own future work calls for this) |
| Gao et al. (2506.09713) | **Empirical** bug taxonomy | Motivation + bug seeds; not an automated detection method |

**Defense vs "you just applied a known method":** (a) a derived PMR catalog for
serving optimizations; (b) three LLM-serving-specific methodological problems &
solutions (noise, nondeterminism, regime-conditioning); (c) real confirmed bugs

- fix PRs.

## 4. PMR Catalog v1

Each PMR = `(optimization/knob, transformation f, relation R on metric M, regime predicate G)`.
Violation counted **only when G holds** (the regime where the promise must hold).

**Family A — Optimization Dominance** (enabling an output-preserving optimization
must not worsen its target metric, within regime):

- A1 Prefix caching: G = shared-prefix ratio ρ>τ ⟹ TTFT(cache on) ≤ TTFT(cache off)+ε; hit_rate>0.
- A2 Speculative decoding: G = acceptance-friendly workload ⟹ throughput(spec on) ≥ throughput(off).
- A3 CUDA graph: throughput(on) ≥ throughput(off) on decode-heavy regime.

**Family B — Monotonicity** (more resource / more reuse opportunity ⟹ non-worsening):

- B1 Cache-capacity monotonicity: hit_rate non-decreasing in gpu_memory_utilization.
- B2 Shared-prefix monotonicity: ↑ρ ⟹ hit_rate non-decreasing, TTFT non-increasing. (#31920/#36493 = violation: ρ=1 yet hit_rate≈0.)
- B3 Concurrency: throughput non-decreasing in max_num_seqs up to saturation.

**Family C — Stability / Equivalence**:

- C1 Temporal stability: repeated identical workload ⟹ throughput stable (no monotonic decay = no leak/regression).
- C2 Order invariance: permuting arrival order of the same request set (fixed concurrency) ⟹ aggregate throughput within ε (no HoL pathology).

**Family D — Cross-version regression**:

- D1 A newer vLLM version must not regress throughput/hit_rate on a fixed workload.

## 5. Methodology — the three LLM-serving challenges

**Reused from Troya:** PMR form `R(M(x), M(f(x)))`; **repeated executions + statistical
test** (not single-run); **seeded-fault validation** to prove PMR sensitivity.

**C1. Performance noise.** N repetitions per config; pin GPU clocks (`nvidia-smi
-lgc`), warmup, single-GPU isolation, fixed seeds. Calibrate ε from run-to-run
variance of an identical config; report a violation only if the difference is
**statistically significant (bootstrap CI / Mann–Whitney U)** AND exceeds ε.

**C2. Nondeterminism.** Core PMRs use **performance** metrics, so output
nondeterminism does not confound them. Residual scheduling nondeterminism is
controlled by seeds + concurrency control + repetition.

**C3. Regime-conditioning (the key intellectual step).** An optimization may
legitimately help some workloads and hurt others. Each PMR carries an explicit
**regime predicate G**, derived from the optimization's documented intended use.
Violations are counted only within G. This prevents false positives from
legitimate trade-offs and is a core contribution.

## 6. Tool Architecture (maximize reuse)

```
PMR spec (decl.) ─┐
workload spec  ───┼─▶ Workload generator ─▶ Runner ─▶ Stat checker ─▶ Triage/report
                  │   (vLLM dataset fwk +   (vllm serve +  (CI/effect   (localize +
                  │    stateful interleave)  vllm bench*)   size test)    reproducer)
```

`*` = our hit-rate-extended `vllm bench serve` (already built).

| Need | Reuse (don't build) | Build new (thin) |
|---|---|---|
| Run + metrics | `vllm bench serve` + our prefix-cache hit-rate fields | orchestration over config families |
| PMR concept + stats + validation | Troya PMR framework, seeded faults | serving PMR catalog + regime predicates |
| Motivation / bug seeds | Gao et al. taxonomy | — |
| Workloads | vLLM dataset framework (sharegpt/random/prefix-len) | stateful arrival/interleave generator (AFLNet-style) |
| Paper template | DLSmith | — |

## 7. Evaluation Plan

- **RQ1 (validity/sensitivity):** seeded-fault detection rate — inject synthetic
  regressions (artificial cache-lookup delay, disabled cache path) → do PMRs catch
  them? (de-risks the empirical concern; result even if zero real bugs.)
- **RQ2 (effectiveness):** real performance bugs found in vLLM across configs/
  versions, confirmed by maintainers / matching known issues → headline + PRs.
- **RQ3 (generality):** apply to SGLang and across vLLM versions → cross-engine /
  cross-version regressions.
- **RQ4 (cost):** runtime, #configs, overhead.
- **Baselines:** vLLM's existing perf tests / ad-hoc benchmarks (catch ~none);
  random config testing.

## 8. Timeline (~12 weeks, solo, parallel with internship)

- W1–2: deep related work; finalize PMR catalog; tool skeleton on `vllm bench`.
- W3–5: runner + statistical checker + workload generator; seeded-fault validation (RQ1).
- W6–9: real-bug hunting across configs/versions (RQ2); triage; file PRs.
- W9–10: generality — SGLang / vLLM versions (RQ3).
- W10–12: writing, artifact packaging, submission.

## 9. Risks & Mitigations

- **R1 few real bugs** → seeded-fault RQ1 still yields a result; strong prior
  (existing issues); optimization-*interaction* space is under-tested.
- **R2 noise swamps signal** → clock pinning, isolation, repetition, calibrated ε,
  significance tests.
- **R3 novelty challenged** → regime-conditioning + LLM-specific challenges + real
  bugs + DLSmith precedent.
- **R4 solo scope** → heavy reuse (vllm bench, Troya scaffold), bounded catalog.

## 10. Target Venues

CCF-B: ICST, ISSRE, SANER, ICSME (+ tool/artifact tracks). Stretch CCF-A:
ISSTA / ASE / FSE if RQ2 results are strong.
