"""Fitting the model from labelled development traces (Section 4.3)."""
import numpy as np
import pytest

from belay.fit import fit, shadow_traces
from belay.model import Check, Job, RunModel
from belay.simulate import World

CRITIC = Check("critic", cost=0.6, recall0=0.65, recall_inf=0.30, decay=0.6, false_flag=0.06)
RULE = Check("rule", cost=0.0, fixed_cost=0.02, recall0=0.95, false_flag=0.01, coverage=(0.0, 1.0, 0.5), after=(1, 2))
TRUE = RunModel(hazard=(0.01, 0.06, 0.03), checks=(CRITIC, RULE), frailty_var=0.8, retry_fresh=0.85, retry_inplace=1.9)


def jobs(count=200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        n = int(rng.integers(15, 40))
        out.append(Job(types=rng.integers(0, 3, n), costs=np.ones(n), irreversible=np.zeros(n, bool)))
    return out


@pytest.fixture(scope="module")
def traces():
    js = jobs()
    return js, shadow_traces(js, World(TRUE), runs_per_job=12, rng=np.random.default_rng(1))


def test_hazards_and_frailty_are_recovered(traces):
    js, tr = traces
    model = fit(js, tr, template=TRUE)
    assert np.allclose(model.hazard, TRUE.hazard, rtol=0.2)
    assert model.frailty_var == pytest.approx(0.8, rel=0.25)


def test_recall_by_latency_and_coverage_are_recovered(traces):
    js, tr = traces
    critic, rule = fit(js, tr, template=TRUE).checks
    assert critic.recall0 == pytest.approx(0.65, abs=0.05)
    assert critic.r_inf == pytest.approx(0.30, abs=0.06)
    assert critic.false_flag == pytest.approx(0.06, abs=0.01)
    assert rule.recall0 == pytest.approx(0.95, abs=0.04)
    assert rule.coverage[0] == pytest.approx(0.0, abs=0.05)
    assert rule.coverage[2] == pytest.approx(0.5, abs=0.08)


def test_retry_multipliers_separate_contamination_from_selection(traces):
    js, tr = traces
    model = fit(js, tr, template=TRUE)
    assert model.retry_fresh == pytest.approx(0.85, rel=0.2)
    assert model.retry_inplace == pytest.approx(1.9, rel=0.2)


def test_the_independence_fit_has_no_frailty_and_flat_recall(traces):
    js, tr = traces
    model = fit(js, tr, template=TRUE, frailty=False, latency=False, retry_modes=False)
    assert model.frailty_var == 0.0
    assert model.retry_fresh == model.retry_inplace == 1.0
    assert all(c.r_inf == c.recall0 for c in model.checks)
