import numpy as np, pytest
from batchsim.cost import Cost, measured, with_ratio
from batchsim.sim import Req, run, make_workload, summarize, wpercentile

C = Cost(alpha=0.001, beta=0.0002, gamma=0.00002)

def test_cost_fit_reproduces_measured_decode_steps():
    c = measured()
    for b, ms in ((1, 0.956), (4, 1.34), (16, 2.826)):
        assert abs(1000 * c.step(b) - ms) < 0.03

def test_single_request_latency_is_exact_for_every_scheduler():
    for sched, budget in (("static", None), ("continuous", None), ("chunked", 10_000)):
        reqs, *_ = run([Req(5.0, 100, 10)], sched, C, B=4, budget=budget)
        r = reqs[0]
        assert r.first - 5.0 == pytest.approx(C.step(0, 100))
        assert r.finish - r.first == pytest.approx(9 * C.step(1))

def test_n_equals_one_request_finishes_at_prefill():
    for sched in ("static", "continuous"):
        r, *_ = run([Req(0.0, 50, 1)], sched, C)
        assert r[0].finish == r[0].first

@pytest.mark.parametrize("sched", ["static", "continuous"])
def test_b1_matches_md1_queue_formula(sched):
    """One slot, deterministic service S, Poisson arrivals: mean wait Wq = rho*S / (2(1-rho))."""
    P, N = 100, 20
    S = C.step(0, P) + (N - 1) * C.step(1); rho = 0.7
    rng = np.random.default_rng(1)
    waits = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        reqs = make_workload(rho / S, 4000 * S, rng, out_mean=N, out_cv=0, in_mean=P, in_cv=0)
        done, *_ = run(reqs, sched, C, B=1)
        waits += [r.first - r.arr - C.step(0, P) for r in done[200:]]
    theory = rho * S / (2 * (1 - rho))
    assert abs(np.mean(waits) - theory) / theory < 0.10

def test_tokens_are_conserved():
    rng = np.random.default_rng(0); reqs = make_workload(30, 5, rng)
    for sched, budget in (("static", None), ("continuous", None), ("chunked", 256)):
        done, iters, _ = run(reqs, sched, C, B=8, budget=budget)
        assert all(r.finish is not None and r.finish >= r.first >= r.arr for r in done)
        assert sum(w for _, w in iters) + len(done) == sum(r.N for r in done)    # decoder tokens + one first token each

def test_uniform_lengths_static_has_no_padding_waste_and_equals_continuous():
    reqs = [Req(0.0, 64, 30) for _ in range(12)]
    s, si, sw = run(reqs, "static", C, B=4); c, ci, cw = run(reqs, "continuous", C, B=4)
    assert sw == 0 and max(r.finish for r in s) == pytest.approx(max(r.finish for r in c))

def test_variable_lengths_static_wastes_and_loses_at_saturation():
    rng = np.random.default_rng(3); reqs = make_workload(1, 1, rng)
    reqs = [Req(0.0, 64, int(n)) for n in rng.integers(1, 400, 96)]
    s, si, sw = run(reqs, "static", C, B=8); c, ci, cw = run(reqs, "continuous", C, B=8)
    assert sw > 0 and cw == 0
    assert max(r.finish for r in c) < max(r.finish for r in s)

def test_batch_cap_is_respected():
    rng = np.random.default_rng(2); reqs = make_workload(200, 2, rng)
    _, iters, _ = run(reqs, "continuous", C, B=3)
    assert max(w for _, w in iters) <= 3

def test_chunked_budget_bounds_every_iteration_and_covers_all_prompt_tokens():
    unit = Cost(alpha=0.0, beta=0.0, gamma=1.0)       # iteration duration == prefill tokens processed
    reqs = [Req(0.0, 1000, 5), Req(0.0, 700, 5), Req(0.0, 40, 5)]
    _, iters, _ = run(reqs, "chunked", unit, B=8, budget=256)
    assert max(d for d, _ in iters) <= 256 + 1e-9
    assert sum(d for d, _ in iters) == pytest.approx(1000 + 700 + 40)

def test_chunking_reduces_worst_inter_token_gap_when_long_prompt_arrives():
    reqs = [Req(0.0, 32, 400)] + [Req(0.1, 4000, 8)]
    _, i1, _ = run(reqs, "continuous", C, B=4); _, i2, _ = run(reqs, "chunked", C, B=4, budget=256)
    gaps = lambda it: max(d for d, w in it if w > 0)
    assert gaps(i2) < gaps(i1)

def test_workload_is_reproducible_and_lengths_have_the_requested_mean():
    a = make_workload(50, 20, np.random.default_rng(5)); b = make_workload(50, 20, np.random.default_rng(5))
    assert [(r.arr, r.P, r.N) for r in a] == [(r.arr, r.P, r.N) for r in b]
    big = make_workload(500, 100, np.random.default_rng(1), out_mean=128, out_cv=1.0)
    assert abs(np.mean([r.N for r in big]) - 128) / 128 < 0.08       # cap at 1024 trims the tail slightly

def test_weighted_percentile_hand_case():
    assert wpercentile([1, 2, 3], [1, 1, 8], 0.5) == 3.0
    assert wpercentile([1, 2, 3], [8, 1, 1], 0.5) == 1.0
