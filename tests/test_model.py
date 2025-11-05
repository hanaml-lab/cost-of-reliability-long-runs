"""One segment of a run: steps a+1..b executed, then one check at b (Section 3.1 of the report)."""
import math

import numpy as np
import pytest

from belay.model import Check, Job, RunModel, frailty_nodes, segment_outcome

PERFECT = Check("perfect", cost=0.5, recall0=1.0)


def chain(n, irreversible=()):
    return Job(types=np.zeros(n, int), costs=np.ones(n), irreversible=np.isin(np.arange(1, n + 1), irreversible))


def test_a_perfect_check_with_one_fresh_retry():
    # Two steps at hazard 0.1: p = exp(-0.2). Success p + (1 - p) p; cost (2 + 0.5) + (1 - p)(2 + 0.5 + 0.35).
    model = RunModel(hazard=(0.1,), checks=(PERFECT,), max_retries=1, rollback_cost=0.35)
    success, cost = segment_outcome(chain(2), model, a=0, b=2, check=0)
    p = math.exp(-0.2)
    assert success == pytest.approx(p + (1 - p) * p)
    assert cost == pytest.approx(2.5 + (1 - p) * 2.85)


def test_recall_depends_on_how_long_ago_the_error_happened():
    # recall(d) = 0.2 - (0.2 - 0.6) 0.5^d: 0.6 for an error at the checked step, 0.4 one step back.
    check = Check("critic", cost=0.5, recall0=0.6, recall_inf=0.2, decay=0.5)
    model = RunModel(hazard=(0.1,), checks=(check,), max_retries=1)
    success, _ = segment_outcome(chain(2), model, a=0, b=2, check=0)
    p = math.exp(-0.2)
    detected = (1 - math.exp(-0.1)) * 0.4 + math.exp(-0.1) * (1 - math.exp(-0.1)) * 0.6
    assert success == pytest.approx(p + detected * p)


def test_errors_before_an_irreversible_step_cannot_be_rolled_back():
    # Step 2 commits a side effect. An error at step 1 or 2, caught at step 3, is fatal; the retry repeats step 3 only.
    model = RunModel(hazard=(0.1,), checks=(PERFECT,), max_retries=1, rollback_cost=0.0)
    success, cost = segment_outcome(chain(3, irreversible=(2,)), model, a=0, b=3, check=0)
    p3, p1 = math.exp(-0.3), math.exp(-0.1)
    caught_late = math.exp(-0.2) * (1 - p1)  # only an error at step 3 is recoverable
    assert success == pytest.approx(p3 + caught_late * p1)
    assert cost == pytest.approx(3.5 + caught_late * 1.5)


def test_a_check_placed_on_the_irreversible_step_runs_before_it_commits():
    model = RunModel(hazard=(0.1,), checks=(PERFECT,), max_retries=1, rollback_cost=0.0)
    success, _ = segment_outcome(chain(2, irreversible=(2,)), model, a=0, b=2, check=0)
    p = math.exp(-0.2)
    assert success == pytest.approx(p + (1 - p) * p)


def test_frailty_scales_the_hazard():
    model = RunModel(hazard=(0.1,), checks=(PERFECT,), max_retries=0)
    success, _ = segment_outcome(chain(2), model, a=0, b=2, check=0, z=3.0)
    assert success == pytest.approx(math.exp(-0.6))


def test_frailty_nodes_have_unit_mean_and_the_requested_variance():
    z, w = frailty_nodes(0.8, 64)
    assert w.sum() == pytest.approx(1.0)
    assert (w * z).sum() == pytest.approx(1.0, abs=0.02)
    assert (w * (z - 1) ** 2).sum() == pytest.approx(0.8, rel=0.1)
    z0, w0 = frailty_nodes(0.0, 16)
    assert z0.tolist() == [1.0] and w0.tolist() == [1.0]
