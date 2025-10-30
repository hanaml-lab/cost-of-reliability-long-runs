"""Closed-form results used in the report.

All of them concern one step, or a homogeneous chain of steps, and are the building blocks the
placement program (``belay.placement``) generalises:

* ``check_gain`` / ``break_even_recall``: when a check with one retry raises a step's success
  (Proposition 1 of the report).
* ``retry_success``: first-attempt and retry success under gamma frailty; a retry after a failure
  is drawn from the harder tasks, so it succeeds less often even with no contamination
  (Proposition 2).
* ``young_interval``: the cost-optimal spacing of checks on a homogeneous chain, Young's
  checkpoint formula with the check in the role of the checkpoint (Appendix B).
* ``half_life_multiplier``: how far checking with recall ``r`` and perfect retries stretches a
  run's reliability half-life (Appendix B).
"""
from __future__ import annotations

import math


def check_gain(p: float, r: float, f: float, q: float) -> float:
    """Change in a step's success probability from adding a check with one retry.

    p: success of the step; r: the check's recall on real errors; f: its false-flag rate;
    q: success of the retry. With the check the step ends correct with probability
    p(1 - f + f q) + (1 - p) r q.
    """
    with_check = p * (1 - f + f * q) + (1 - p) * r * q
    return with_check - p


def break_even_recall(p: float, f: float, q: float) -> float:
    """Recall above which the check helps: (1 - p) r q > p f (1 - q). May exceed one."""
    if q <= 0:
        return math.inf
    return p * f * (1 - q) / ((1 - p) * q)


def _gamma_laplace(s: float, frailty_var: float) -> float:
    """E[exp(-s Z)] for Z ~ Gamma with mean 1 and the given variance (a point mass at 1 if zero)."""
    if frailty_var <= 0:
        return math.exp(-s)
    return (1 + frailty_var * s) ** (-1 / frailty_var)


def retry_success(eta: float, frailty_var: float, rho: float = 1.0) -> tuple[float, float]:
    """First-attempt success and the success of a retry after a failure.

    The step derails with probability 1 - exp(-Z eta) for a task with frailty Z; the retry's hazard
    is multiplied by rho (above one when the failed attempt stays in context). Returns
    (P(first ok), P(retry ok | first failed)).
    """
    first = _gamma_laplace(eta, frailty_var)
    both = _gamma_laplace(rho * eta, frailty_var) - _gamma_laplace((1 + rho) * eta, frailty_var)
    return first, both / (1 - first)


def young_interval(check_cost: float, hazard: float, step_cost: float) -> float:
    """Steps between checks that minimise check cost plus expected rework: sqrt(2 k / (h c))."""
    return math.sqrt(2 * check_cost / (hazard * step_cost))


def half_life_multiplier(hazard: float, recall: float) -> float:
    """Ratio of a run's half-life with a check after every step to its half-life without.

    With unlimited, independent retries a step ends correct with probability (1 - h) / (1 - h r),
    so the half-life grows by log(1 - h) / log((1 - h) / (1 - h r)), which tends to 1 / (1 - r)
    as h -> 0.
    """
    survive = (1 - hazard) / (1 - hazard * recall)
    return math.log(1 - hazard) / math.log(survive)
