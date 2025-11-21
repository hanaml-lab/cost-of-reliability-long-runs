"""The adaptive policy: re-plan from the frailty posterior after each flag (Section 4.2)."""
import numpy as np
import pytest

from belay.adaptive import adaptive_plans, posterior_mean
from belay.model import Check, Job, RunModel

CRITIC = Check("critic", cost=0.6, recall0=0.65, recall_inf=0.3, decay=0.6, false_flag=0.06)


def test_the_grid_posterior_matches_the_conjugate_gamma_update():
    # Prior Gamma(shape 2, rate 2) (variance 0.5); exposure 1.5 and 2 detections: posterior mean (2 + 2) / (2 + 1.5).
    assert posterior_mean(frailty_var=0.5, exposure=1.5, detections=2, q=400) == pytest.approx(4 / 3.5, rel=0.02)


def test_flags_make_the_policy_check_more_often():
    job = Job(types=np.zeros(30, int), costs=np.ones(30), irreversible=np.zeros(30, bool))
    model = RunModel(hazard=(0.03,), checks=(CRITIC,), frailty_var=1.0, max_retries=2)
    plans = adaptive_plans(job, model, lam=0.02, max_detections=3, q=12)
    counts = [len(p.checks) for p in plans]
    assert counts == sorted(counts) and counts[-1] > counts[0]
