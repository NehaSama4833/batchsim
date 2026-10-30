"""Single-replica batching schedulers on an iteration cost model. All three share one cost model and one
workload, differ only in WHEN requests may join the batch:
  static     : take up to B queued requests, prefill them together, decode until the LAST one finishes
               (finished slots keep costing compute: padding), then take the next batch.
  continuous : iteration-level scheduling (Orca-style). Each iteration admits queued requests into free slots;
               finished sequences leave immediately. A new request's whole prompt is prefilled in one iteration.
  chunked    : continuous + a per-iteration token budget: decoding sequences count 1 token each, the rest of the
               budget goes to prompt chunks (Sarathi/vLLM chunked prefill), bounding how long decodes stall.
No KV-cache memory limit: only a cap B on concurrent sequences."""
from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class Req:
    arr: float
    P: int
    N: int
    first: float = None
    finish: float = None


def make_workload(rate, duration, rng, out_mean=128, out_cv=1.0, in_mean=256, in_cv=0.5, long_frac=0.0, long_len=4096,
                  out_cap=1024, in_cap=2048):
    def logn(mean, cv, n, cap, lo=1):
        if cv == 0: return np.full(n, int(min(mean, cap)))
        s2 = np.log(1 + cv * cv); mu = np.log(mean) - s2 / 2
        return np.clip(np.rint(rng.lognormal(mu, np.sqrt(s2), n)), lo, cap).astype(int)
    t, arr = 0.0, []
    while True:
        t += rng.exponential(1 / rate)
        if t >= duration: break
        arr.append(t)
    n = len(arr)
    P = logn(in_mean, in_cv, n, in_cap); N = logn(out_mean, out_cv, n, out_cap)
    if long_frac:
        P = np.where(rng.random(n) < long_frac, long_len, P)
    return [Req(float(a), int(p), int(m)) for a, p, m in zip(arr, P, N)]


def _fresh(reqs):
    return [Req(r.arr, r.P, r.N) for r in reqs]


def run(reqs, sched, cost, B=32, budget=None):
    """Simulate one replica. Returns (requests with first/finish set, iterations, wasted_slot_steps).
    iterations: list of (duration, tokens_emitted_by_decoders). A sequence emits one token per iteration while
    decoding, so the inter-token gap it sees in an iteration equals that iteration's duration."""
    reqs = _fresh(reqs)
    reqs.sort(key=lambda r: r.arr)
    n = len(reqs)
    iters, waste = [], 0
    t, i, waiting = 0.0, 0, deque()
    if sched == "static":
        while i < n or waiting:
            if not waiting and reqs[i].arr > t: t = reqs[i].arr
            while i < n and reqs[i].arr <= t: waiting.append(reqs[i]); i += 1
            batch = [waiting.popleft() for _ in range(min(B, len(waiting)))]
            dur = cost.step(0, sum(r.P for r in batch))          # batched prefill; first tokens appear at its end
            t += dur; iters.append((dur, 0))
            for r in batch:
                r.first = t
                if r.N == 1: r.finish = t
            b0 = len(batch)
            steps = max(r.N for r in batch) - 1
            for k in range(1, steps + 1):
                emitting = [r for r in batch if r.N - 1 >= k]     # unfinished before this step
                dur = cost.step(b0)                               # padding: full batch size is paid every step
                t += dur; iters.append((dur, len(emitting))); waste += b0 - len(emitting)
                for r in emitting:
                    if r.N - 1 == k: r.finish = t
        return reqs, iters, waste
    running, prefilling = [], []                 # running: [req, decode steps left]; prefilling: [req, prompt tokens left]
    while i < n or waiting or running or prefilling:
        if not (waiting or running or prefilling) and reqs[i].arr > t: t = reqs[i].arr
        while i < n and reqs[i].arr <= t: waiting.append(reqs[i]); i += 1
        while waiting and len(running) + len(prefilling) < B:
            r = waiting.popleft(); prefilling.append([r, r.P])
        dec = running
        room = float("inf") if budget is None else max(0, budget - len(dec))
        c, finished_pf = 0, []
        for pf in prefilling:
            take = min(pf[1], room - c)
            if take <= 0: break
            pf[1] -= take; c += take
            if pf[1] == 0: finished_pf.append(pf)
        dur = cost.step(len(dec), c)
        t += dur; iters.append((dur, len(dec)))
        nxt = []
        for item in dec:
            item[1] -= 1
            if item[1] == 0: item[0].finish = t
            else: nxt.append(item)
        for pf in finished_pf:
            prefilling.remove(pf); r = pf[0]; r.first = t
            if r.N == 1: r.finish = t
            else: nxt.append([r, r.N - 1])
        running = nxt
    return reqs, iters, waste


def wpercentile(values, weights, q):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    o = np.argsort(v); v, w = v[o], w[o]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, q * c[-1])]) if c[-1] > 0 else float("nan")


def summarize(reqs, iters, waste, duration, warmup=0.1):
    """Latency stats over requests that arrived after the warm-up. ITL is weighted by tokens emitted."""
    done = [r for r in reqs if r.arr >= warmup * duration]
    ttft = np.array([r.first - r.arr for r in done]); e2e = np.array([r.finish - r.arr for r in done])
    tpot = np.array([(r.finish - r.first) / (r.N - 1) for r in done if r.N > 1])
    dur = np.array([d for d, w in iters if w > 0]); wt = np.array([w for d, w in iters if w > 0])
    end = max(r.finish for r in reqs)
    toks = sum(r.N for r in reqs)
    return dict(n=len(done), tput=toks / end, drain_s=max(0.0, end - duration),
                ttft_p50=float(np.quantile(ttft, .5)), ttft_p99=float(np.quantile(ttft, .99)),
                e2e_p50=float(np.quantile(e2e, .5)), e2e_p99=float(np.quantile(e2e, .99)),
                tpot_p50=float(np.quantile(tpot, .5)), tpot_p99=float(np.quantile(tpot, .99)),
                itl_p99=wpercentile(dur, wt, .99), itl_p999=wpercentile(dur, wt, .999), itl_max=float(dur.max()), waste_frac=waste / max(1, waste + sum(w for _, w in iters)))
