import argparse, json, time
import numpy as np
from .cost import measured, gpu_like, with_ratio
from .sim import make_workload, run, summarize

KEYS = ["tput", "drain_s", "ttft_p50", "ttft_p99", "e2e_p50", "e2e_p99", "tpot_p50", "tpot_p99", "itl_p99", "itl_p999", "itl_max", "waste_frac"]


def capacity(cost, B=32, sched="continuous", budget=None, **wl):
    """Saturated output rate (tokens/s) of a scheduler: a large backlog all present at t=0."""
    rng = np.random.default_rng(999)
    reqs = make_workload(1e6, 0.003, rng, **wl)          # ~3000 simultaneous arrivals
    done, iters, w = run(reqs, sched, cost, B=B, budget=budget)
    return sum(r.N for r in done) / max(r.finish for r in done)


def cell(cost, sched, rho, seeds, duration, B=32, budget=None, cap_sched="continuous", **wl):
    cap = capacity(cost, B, sched=cap_sched, **wl)
    mean_n = wl.get("out_mean", 128)
    rate = rho * cap / mean_n
    rs = []
    for s in range(seeds):
        reqs = make_workload(rate, duration, np.random.default_rng(s), **wl)
        done, iters, waste = run(reqs, sched, cost, B=B, budget=budget)
        rs.append(summarize(done, iters, waste, duration))
    out = {k: float(np.mean([r[k] for r in rs])) for k in KEYS}
    out["e2e_p99_range"] = [min(r["e2e_p99"] for r in rs), max(r["e2e_p99"] for r in rs)]
    out["capacity_tok_s"] = cap
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--out", default="results/experiment.json")
    a = ap.parse_args(argv)
    cost = measured(); t0 = time.time()
    out = {"cost": cost.__dict__, "seeds": a.seeds, "duration": a.duration, "load": [], "load_static": [], "outcv": [], "capratio_cv": [], "capratio_shape": [], "longprompt": [], "shape": []}
    for rho in (0.3, 0.5, 0.7, 0.85, 0.95):
        for name, sc, bud in (("static", "static", None), ("continuous", "continuous", None), ("chunked512", "chunked", 512)):
            out["load"].append(dict(rho=rho, sched=name, **cell(cost, sc, rho, a.seeds, a.duration, budget=bud)))
    for rho in (0.3, 0.6, 0.9):       # offered load as a fraction of STATIC's own saturated capacity: both schedulers stable
        for name, sc, bud in (("static", "static", None), ("continuous", "continuous", None)):
            out["load_static"].append(dict(rho=rho, sched=name, **cell(cost, sc, rho, a.seeds, a.duration, budget=bud, cap_sched="static")))
    print("load done %.0fs" % (time.time() - t0), flush=True)
    for cv in (0.0, 0.5, 1.0, 2.0):
        for name, sc in (("static", "static"), ("continuous", "continuous")):
            out["outcv"].append(dict(out_cv=cv, sched=name, **cell(cost, sc, 0.6, a.seeds, a.duration, out_cv=cv)))
    for cv in (0.0, 0.25, 0.5, 1.0, 2.0):
        cs, cc = capacity(cost, sched="static", out_cv=cv), capacity(cost, sched="continuous", out_cv=cv)
        out["capratio_cv"].append(dict(out_cv=cv, static=cs, continuous=cc, ratio=cs / cc))
    print("outcv done %.0fs" % (time.time() - t0), flush=True)
    for name, sc, bud in (("continuous", "continuous", None), ("chunked256", "chunked", 256), ("chunked512", "chunked", 512),
                          ("chunked1024", "chunked", 1024), ("chunked2048", "chunked", 2048)):
        out["longprompt"].append(dict(sched=name, **cell(cost, sc, 0.6, a.seeds, a.duration, budget=bud, long_frac=0.05, long_len=4096, in_cap=4096)))
    print("longprompt done %.0fs" % (time.time() - t0), flush=True)
    shapes = [("measured (beta/alpha=%.2f)" % (cost.beta / cost.alpha), cost)] + \
             [("beta/alpha=%.2f" % r, with_ratio(cost, r)) for r in (0.01, 0.05, 0.5, 1.0)] + [("gpu-like (assumed)", gpu_like())]
    for name, cm in shapes:
        for sc in ("static", "continuous"):
            out["shape"].append(dict(shape=name, sched=sc, **cell(cm, sc, 0.6, a.seeds, a.duration)))
    for name, cm in shapes:
        cs, cc = capacity(cm, sched="static"), capacity(cm, sched="continuous")
        out["capratio_shape"].append(dict(shape=name, static=cs, continuous=cc, ratio=cs / cc))
    print("shape done %.0fs" % (time.time() - t0), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
