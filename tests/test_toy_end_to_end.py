"""The CPU toy setting runs the whole pipeline: shadow traces, fit, plan, simulate."""
import time

from belay.toy import run_toy


def test_the_toy_pipeline_beats_uniform_checking_at_lower_cost():
    start = time.time()
    out = run_toy(seed=0)
    assert time.time() - start < 60
    assert out["belay"]["success"] > out["none"]["success"] + 0.08
    assert out["belay"]["tokens"] < out["every"]["tokens"]
    assert out["belay"]["success"] >= out["every"]["success"] - 0.02
    assert out["belay"]["success"] > out["indep"]["success"]
