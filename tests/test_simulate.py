"""The Monte Carlo run simulator: the ground truth the report's tables are measured on."""
import math

import numpy as np
import pytest
from scipy import integrate, stats

from belay.model import Check, Job, RunModel
from belay.placement import solve
from belay.simulate import World, every_step, no_checks, periodic, simulate, static, voting

CRITIC = Check("critic", cost=0.6, recall0=0.65, recall_inf=0.3, decay=0.6, false_flag=0.06)
RULE = Check("rule", cost=0.0, fixed_cost=0.02, recall0=0.97, false_flag=0.005, coverage=(0.0, 1.0), after=(1,))


def job(n=24, seed=0):
    rng = np.random.default_rng(seed)
    types = rng.integers(0, 2, n)
    irreversible = np.zeros(n, bool)
    irreversible[[n // 3, 2 * n // 3]] = True
    return Job(types=types, costs=np.linspace(1.0, 1.5, n), irreversible=irreversible)


def test_simulated_success_matches_the_program_when_the_world_is_the_model():
    model = RunModel(hazard=(0.02, 0.08), checks=(CRITIC, RULE), max_retries=2)
    j = job()
    plan = solve(j, model, lam=0.01)
    out = simulate(j, World(model), static(plan), runs=40000, rng=np.random.default_rng(1))
    se = math.sqrt(plan.success * (1 - plan.success) / 40000)
    assert out.success.mean() == pytest.approx(plan.success, abs=4 * se)
    # The program prices segments for runs still on track; retries spent on runs already lost come on top.
    assert plan.cost < out.tokens.mean() < 1.05 * plan.cost


def test_without_checks_success_is_the_frailty_average_of_the_survival():
    model = RunModel(hazard=(0.02, 0.08), checks=(CRITIC,))
    j = job()
    H = model.eta(j).sum()
    var = 0.9
    s2 = math.log(1 + var)
    expected, _ = integrate.quad(lambda z: math.exp(-z * H) * stats.lognorm.pdf(z, s=math.sqrt(s2), scale=math.exp(-s2 / 2)), 0, np.inf)
    out = simulate(j, World(model, frailty="lognormal", frailty_var=var), no_checks(), runs=40000, rng=np.random.default_rng(2))
    assert out.success.mean() == pytest.approx(expected, abs=0.01)
    assert out.tokens.mean() == pytest.approx(j.costs.sum())


def test_voting_with_fully_shared_mistakes_is_no_better_than_one_sample():
    model = RunModel(hazard=(0.02, 0.08), checks=(CRITIC,))
    j = job()
    single = simulate(j, World(model), no_checks(), runs=40000, rng=np.random.default_rng(3)).success.mean()
    shared = simulate(j, World(model, vote_corr=1.0), voting(3), runs=40000, rng=np.random.default_rng(4))
    independent = simulate(j, World(model, vote_corr=0.0), voting(3), runs=40000, rng=np.random.default_rng(5))
    assert shared.success.mean() == pytest.approx(single, abs=0.012)
    assert independent.success.mean() > single + 0.2
    assert shared.tokens.mean() == pytest.approx(3.1 * j.costs.sum())


def test_retrying_with_the_failed_attempt_in_context_does_worse_than_rolling_back():
    model = RunModel(hazard=(0.02, 0.08), checks=(CRITIC, RULE), retry_fresh=0.85, retry_inplace=1.9, max_retries=2)
    j = job()
    inplace = simulate(j, World(model), every_step(0, mode="inplace"), runs=20000, rng=np.random.default_rng(6))
    fresh = simulate(j, World(model), every_step(0, mode="fresh"), runs=20000, rng=np.random.default_rng(6))
    assert fresh.success.mean() > inplace.success.mean() + 0.02


def test_periodic_checks_land_every_t_steps():
    model = RunModel(hazard=(0.0, 0.0), checks=(Check("exact", cost=0.5, recall0=1.0),))
    out = simulate(job(n=24), World(model), periodic(0, every=5), runs=10, rng=np.random.default_rng(7))
    assert out.checks.tolist() == [4] * 10  # steps 5, 10, 15, 20; the last four run unchecked
