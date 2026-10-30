"""Iteration cost model:  t = alpha + beta * (decoding sequences) + gamma * (prefill tokens),  in seconds.
`measured()` fits it to Quantbench's real timings (a 3.4M-parameter model on 2 CPU cores): decode step
time at batch 1/4/16 and prefill time of 64 tokens x batch 1/4/16. `gpu_like()` is an ASSUMED shape (large
fixed cost, nearly flat in batch size), not a measurement."""
import json, os
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Cost:
    alpha: float
    beta: float
    gamma: float

    def step(self, decoding, prefill_tokens=0):
        return self.alpha + self.beta * decoding + self.gamma * prefill_tokens


def measured(path=None):
    path = path or os.path.join(os.path.dirname(__file__), "..", "..", "results", "quantbench_speed.json")
    sp = json.load(open(path))["speed"]
    dec = [s for s in sp if s["cache"]]
    b = np.array([s["batch"] for s in dec], float); t = np.array([s["step_ms"] for s in dec]) / 1000
    beta, alpha = np.polyfit(b, t, 1)
    T = np.array([s["batch"] * 64 for s in dec], float); p = np.array([s["prefill_ms"] for s in dec]) / 1000
    gamma = float(np.sum((p - alpha) * T) / np.sum(T * T))      # least squares through alpha, slope only
    return Cost(float(alpha), float(beta), gamma)


def gpu_like():
    return Cost(alpha=0.025, beta=0.00015, gamma=0.0001)      # ASSUMED: 25 ms fixed, +0.15 ms/seq, 0.1 ms/prompt token


def with_ratio(base: Cost, ratio: float):
    """Keep alpha and gamma, set beta = ratio * alpha (how steeply step time grows with batch size)."""
    return Cost(base.alpha, ratio * base.alpha, base.gamma)
