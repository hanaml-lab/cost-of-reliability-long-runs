"""Algorithm 1: the Lagrangian placement program against brute-force enumeration."""
import itertools
import math

import numpy as np
import pytest

from belay.model import Check, Job, RunModel, frailty_nodes, segment_outcome
from belay.placement import solve

CRITIC = Check("critic", cost=0.6, recall0=0.65, recall_inf=0.3, decay=0.6, false_flag=0.06)
RULE = Check("rule", cost=0.0, fixed_cost=0.02, recall0=0.97, false_flag=0.005, coverage=(0.0, 1.0), after=(1,))


def small_job():
    types = np.array([0, 1, 0, 1, 1, 0, 1])
    return Job(types=types, costs=np.linspace(1.0, 1.6, 7), irreversible=np.array([0, 0, 0, 1, 0, 0, 0], bool))


def objective(job, model, plan, lam, q):
    """Sum over segments of log E[S | on track at a] - lam E[C | on track at a], plus the unchecked tail."""
    z, w = frailty_nodes(model.frailty_var, q)
    eta = model.eta(job)
    H = np.concatenate([[0.0], np.cumsum(eta)])
    total, a = 0.0, 0
    for b, v in plan:
        post = w * np.exp(-z * H[a])
        post /= post.sum()
        outs = [segment_outcome(job, model, a, b, v, z=zi) for zi in z]
        total += math.log(sum(pi * s for pi, (s, _) in zip(post, outs))) - lam * sum(pi * c for pi, (_, c) in zip(post, outs))
        a = b
    post = w * np.exp(-z * H[a])
    post /= post.sum()
    total += math.log((post * np.exp(-z * (H[-1] - H[a]))).sum()) - lam * job.costs[a:].sum()
    return total


def brute_force(job, model, lam, q):
    best, best_plan = -math.inf, None
    options = [[None] + [v for v, c in enumerate(model.checks) if c.applies(job.types[b - 1])] for b in range(1, job.n + 1)]
    for choice in itertools.product(*options):
        plan = [(b, v) for b, v in zip(range(1, job.n + 1), choice) if v is not None]
        val = objective(job, model, plan, lam, q)
        if val > best:
            best, best_plan = val, plan
    return best, best_plan


@pytest.mark.parametrize("frailty_var", [0.0, 0.7])
@pytest.mark.parametrize("lam", [0.002, 0.02, 0.08])
def test_the_program_finds_the_best_placement(frailty_var, lam):
    model = RunModel(hazard=(0.03, 0.12), checks=(CRITIC, RULE), frailty_var=frailty_var, max_retries=2)
    job = small_job()
    plan = solve(job, model, lam, q=6)
    best, best_plan = brute_force(job, model, lam, q=6)
    assert plan.objective == pytest.approx(best, abs=1e-9)
    assert plan.checks == best_plan


def test_a_higher_price_on_tokens_buys_less_checking():
    model = RunModel(hazard=(0.03, 0.12), checks=(CRITIC, RULE), frailty_var=0.7)
    job = small_job()
    costs = [solve(job, model, lam, q=6).cost for lam in (0.0005, 0.005, 0.05, 0.5)]
    assert costs == sorted(costs, reverse=True)
    assert solve(job, model, 5.0, q=6).checks == []


def test_on_the_linear_objective_a_job_likely_to_fail_buys_less_checking():
    # The local price is lam0 / (C0 * S*): below S* = 1 it is higher than the log program's lam0 / C0.
    from belay.placement import solve_linear
    hard = RunModel(hazard=(0.15, 0.4), checks=(CRITIC, RULE), frailty_var=0.7)
    job = small_job()
    c0 = job.costs.sum()
    for lam0 in (0.1, 1.0):
        assert solve_linear(job, hard, lam0, q=6).cost <= solve(job, hard, lam0 / c0, q=6).cost
    assert solve_linear(job, hard, 1.0, q=6).cost < solve(job, hard, 1.0 / c0, q=6).cost
