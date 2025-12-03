"""The adaptive policy of the report (Section 4.2).

After a passed check at position a with d flags so far, the frailty posterior is approximately
Gamma(1/var + d, 1/var + H(a)): the exposure H(a) is the cumulative base hazard up to a, and each flag
counts as one event of the Poisson-gamma model. ``belay.placement`` folds this posterior into its
segment tables, so a plan solved with ``detections=d`` is the best continuation from every rollback
point for a run that has been flagged d times. ``adaptive_plans`` solves one plan per d; the simulator
(``belay.simulate.adaptive``) picks the plan for the run's current count, which re-plans the remaining
checks after every flag at the cost of max_detections + 1 solves per job instead of one per run.
"""
from __future__ import annotations

import numpy as np

from .model import Job, RunModel, frailty_nodes
from .placement import Plan, _dp, local_price, segment_tables, solve


def posterior_mean(frailty_var: float, exposure: float, detections: int, q: int = 64) -> float:
    """E[Z | on track through exposure H, d flags] on the quadrature grid the planner uses."""
    z, w = frailty_nodes(frailty_var, q)
    logw = np.log(w) + detections * np.log(z) - exposure * z
    p = np.exp(logw - logw.max())
    return float((p * z).sum() / p.sum())


def adaptive_plans(job: Job, model: RunModel, lam: float, max_detections: int = 3, linear: bool = False, **kwargs) -> list[Plan]:
    """One plan per flag count 0..max_detections; a run with more flags uses the last.

    With ``linear=True`` ``lam`` is the price lam0 of the linear objective; the job's local price is set
    once, from the unflagged plan (``placement.local_price``), so a flag changes the plan only through the
    frailty posterior.
    """
    if not linear:
        return [solve(job, model, lam, detections=d, **kwargs) for d in range(max_detections + 1)]
    tables = [segment_tables(job, model, detections=d, **kwargs) for d in range(max_detections + 1)]
    price = local_price(job, tables[0], lam)
    return [_dp(job, t, price) for t in tables]
