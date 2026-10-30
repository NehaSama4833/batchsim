import json, sys

def f(x, k):
    v = x[k]
    return "%.1f ms" % (1000 * v) if k.startswith("itl") else "%.0f" % v if k == "tput" else ("%.1f%%" % (100 * v) if k == "waste_frac" else "%.3f" % v)

def table(rows, keyname, keys):
    print("%-26s %-12s" % (keyname, "scheduler") + "".join("%-11s" % k for k in keys))
    for r in rows:
        print("%-26s %-12s" % (r.get(keyname.split()[0], r.get("rho", r.get("out_cv", r.get("shape")))), r["sched"]) + "".join("%-11s" % f(r, k) for k in keys))

def main(p):
    d = json.load(open(p)); c = d["cost"]
    print("cost (s): alpha %.5f beta %.6f gamma %.7f | seeds %d | arrivals for %gs" % (c["alpha"], c["beta"], c["gamma"], d["seeds"], d["duration"]))
    ks = ["ttft_p50", "ttft_p99", "e2e_p99", "itl_p99", "waste_frac"]
    print("\nLOAD relative to STATIC's saturated capacity (both schedulers stable)"); table(d["load_static"], "rho", ks)
    print("\nLOAD relative to CONTINUOUS saturated capacity (static is overloaded wherever its TTFT keeps growing)"); table(d["load"], "rho", ks)
    print("\nSATURATED CAPACITY (tokens/s), static vs continuous, by output-length CV:")
    for r in d["capratio_cv"]: print("  cv %.2f  static %5.0f  continuous %5.0f  ratio %.2f" % (r["out_cv"], r["static"], r["continuous"], r["ratio"]))
    print("SATURATED CAPACITY by cost-model shape:")
    for r in d["capratio_shape"]: print("  %-28s static %5.0f  continuous %5.0f  ratio %.2f" % (r["shape"], r["static"], r["continuous"], r["ratio"]))
    print("\nOUTPUT-LENGTH VARIABILITY (CV of output length), rho 0.6"); table(d["outcv"], "out_cv", ["ttft_p99", "e2e_p99", "waste_frac"])
    print("\nLONG PROMPTS (5% of prompts are 4096 tokens), rho 0.6"); table(d["longprompt"], "sched", ["ttft_p99", "e2e_p99", "itl_p99", "itl_p999", "itl_max"])
    print("\nCOST-MODEL SHAPE, rho 0.6 (rho is relative to continuous capacity under each shape)")
    table(d["shape"], "shape", ["ttft_p99", "e2e_p99", "waste_frac"])

if __name__ == "__main__":
    main(sys.argv[1])
