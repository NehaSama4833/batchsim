# Batchsim

A small simulator of **one LLM-serving replica** comparing three batching schedulers on the same workload and cost model:
- **static**: take up to B queued requests, prefill them together, decode until the *last* one finishes (finished slots keep costing compute), then take the next batch;
- **continuous** (Orca-style iteration-level scheduling): requests join free slots every iteration, finished ones leave at once;
- **chunked**: continuous plus a per-iteration token budget so a long prompt cannot stall everyone's decoding (chunked prefill, as in Sarathi/vLLM).

**Status: Day 10 MVP, simulation only.** Not a serving system, not run on a GPU, no KV-cache memory model.

## Problem and what exists
Continuous batching, chunked prefill and paged KV memory are standard in vLLM, TGI, TensorRT-LLM and SGLang; Orca (OSDI '22) introduced iteration-level scheduling. This repo adds nothing to them. It is a small executable model for seeing **how much** scheduling policy matters, and for what workloads, with every assumption visible. It extends Quantbench (Day 6), which measured that batching raises throughput on a real tiny model, and uses that model's measured timings for its cost model. It is different from Routewise (routing *between* replicas) and Scalesim (autoscaling *replica count*).

## Cost model (`cost.py`)
One iteration takes `alpha + beta * (decoding sequences) + gamma * (prompt tokens processed)` seconds. `measured()` least-squares fits this to Quantbench's recorded timings (fp32 3.4M-parameter model, 2 CPU cores): alpha = 0.84 ms, beta = 0.124 ms per sequence, gamma = 0.024 ms per prompt token (beta/alpha = 0.15). The fit reproduces the measured decode step at batch 1/4/16 to within 0.03 ms (a test checks this) and prefill within about 20% (2.35 ms vs 2.89 ms measured at the smallest size). `gpu_like()` is an **assumed** flatter shape, not a measurement. Workload: Poisson arrivals, lognormal output length (mean 128, CV 1, capped at 1024), lognormal prompt length (mean 256, CV 0.5), max 32 concurrent sequences.

## Run
```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q                                          # 13 tests
.venv/bin/python -m batchsim.experiment                       # about 5 s
.venv/bin/python -m batchsim.report results/experiment.json   # tables (also results/experiment_table.txt)
```

## Validation
- One slot, deterministic service, Poisson arrivals: simulated mean queueing delay matches the **M/D/1 formula** `rho*S / (2(1-rho))` within 10% (rho 0.7), for both static and continuous.
- A single request's TTFT and finish time equal prefill + (N-1) steps exactly, for all three schedulers.
- Tokens are conserved (decoder tokens + one first token per request = total output) and no iteration exceeds the batch cap; with a token budget, no iteration processes more than the budget and all prompt tokens are covered once.
- With uniform output lengths the static scheduler has zero padding waste and finishes exactly when continuous does.
There is no comparison against a real serving system.

## Results (3 seeds, 40 s of arrivals, measured cost model; `results/experiment_table.txt`)
TTFT = queue wait + prefill. ITL = gap between a sequence's consecutive tokens (iteration duration weighted by tokens). "Capacity" = saturated output rate with a large backlog.

1. **Output-length variability is what hurts static batching.** Static capacity relative to continuous: 1.00 at uniform lengths, 0.68 (CV 0.25), 0.49 (CV 0.5), 0.30 (CV 1.0, my default), 0.22 (CV 2.0). Short requests wait for the longest in their batch; at CV 1.0 about 75% of decode slot-steps are padding at high load.
2. **At the same offered load, latency differs by orders of magnitude, even below static's capacity.** At 60% of static's capacity: TTFT p99 0.97 s (static) vs 0.018 s (continuous); at 90%: 1.6 s vs 0.019 s; end-to-end p99 1.4 s vs 0.7 s and 2.2 s vs 0.7 s. This compares the schedulers at equal load, so continuous is lightly loaded in these rows; it shows the batch-boundary wait, not a stress test.
3. **Beyond about 30% of continuous capacity, static falls behind.** Its TTFT keeps growing (p99 28 s at 50%, 91 s at 95% of continuous capacity) because arrivals exceed what it can serve. Continuous TTFT p99 stays under 0.8 s at 95%.
4. **Chunked prefill trades a rare big stall for more frequent small ones.** With 5% of prompts at 4096 tokens (60% load): max inter-token gap 121 ms (continuous) vs 9, 15, 27, 51 ms for budgets 256, 512, 1024, 2048; p99.9 99 ms vs 9, 15, 26, 50 ms. But p99 ITL is not better in general: 10.4 ms (continuous) vs 8.2, 13.7, 19.5, 11.6 ms. Chunking makes many iterations slightly longer while a prompt is being fed in. TTFT and end-to-end p99 were unchanged within noise (TTFT p99 0.14 to 0.16 s). Without long prompts chunking made no visible difference.
5. **The cost-model shape did not change the ranking.** Static/continuous capacity ratio across beta/alpha of 0.01, 0.05, 0.15 (measured), 0.5, 1.0 and the assumed GPU-like shape: 0.44, 0.36, 0.30, 0.26, 0.25, 0.29. Static is relatively less bad when batch size barely costs anything (padding is cheap) but never close.

## Limitations (read these)
- **A model, not a server.** The cost model is linear and was fitted to a 3.4M-parameter CPU model; real GPU step times, attention cost growing with context length (quadratic prefill) and kernel behaviour are not captured. The GPU-like row is assumed.
- **No KV-cache memory limit.** In practice memory, not a fixed slot count, bounds concurrency; paged attention, preemption and swapping are not modelled. Continuous batching's real advantage depends on that too.
- **Workload shapes are my choices:** lognormal lengths, Poisson arrivals, FIFO, one replica, and a static scheduler that does not wait to fill a batch. A static scheduler with a batching window or length-bucketed batches would do better than this one.
- **The 30% static/continuous capacity ratio is specific to CV 1.0 outputs capped at 1024.** Do not quote it as a general number; see the CV sweep.
- **Three seeds, no confidence intervals.** The load sweep numbers are close to deterministic at this scale but I did not quantify variance beyond storing e2e p99 ranges in the JSON.
- ITL here is per-iteration, so it does not include network or detokenisation effects; chunked prefill cost is linear in tokens.

## Future work
KV-memory limits with paged allocation and preemption; length-aware or windowed static baselines; prefill/decode disaggregation; priority and SLO-aware admission; calibration on a real GPU or llama.cpp.
