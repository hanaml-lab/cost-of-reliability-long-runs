"""The derailment model of a long agent run (Section 3.1 of the report).

A job is a sequence of n steps, each of a type with a base hazard. A task has a latent frailty Z
(mean 1): step i derails an on-track run with probability 1 - exp(-Z * eta[type_i]). A derailed run
stays wrong until a check catches the error and the run rolls back. Checks have a cost, a false-flag
rate on on-track runs, a coverage per error type, and a recall that depends on how many steps ago the
error happened: recall(d) = recall_inf - (recall_inf - recall0) * decay**d. A step marked
irreversible commits an external side effect when it runs; a check placed *on* that step runs before
the commit, but an error made at or before a committed step can no longer be rolled back.

``segment_outcome`` gives the success probability and expected cost of one segment, steps a+1..b
followed by a check at b, with up to ``max_retries`` retries after a flag. It is the scalar reference
for the vectorised tables in ``belay.placement``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Check:
    """A check that can run after a step.

    cost: tokens, as a multiple of the cost of the step it follows (a critic reads that context);
    fixed_cost: tokens per call regardless of position; recall0/recall_inf/decay: recall on an error
    made at the checked step, on an old error, and how fast one turns into the other; false_flag: the
    probability it flags an on-track run; coverage: per step type, the share of that type's errors it
    can see at all (None: all); after: the step types it can follow (None: any).
    """

    name: str
    cost: float
    recall0: float
    recall_inf: float | None = None
    decay: float = 0.5
    false_flag: float = 0.0
    coverage: tuple[float, ...] | None = None
    after: tuple[int, ...] | None = None
    fixed_cost: float = 0.0

    @property
    def r_inf(self) -> float:
        return self.recall0 if self.recall_inf is None else self.recall_inf

    def recall(self, latency):
        """Recall on an error made ``latency`` steps before the check (0: at the checked step)."""
        return self.r_inf - (self.r_inf - self.recall0) * np.power(self.decay, latency)

    def covers(self, step_type: int) -> float:
        return 1.0 if self.coverage is None else self.coverage[step_type]

    def applies(self, step_type: int) -> bool:
        return self.after is None or step_type in self.after

    def cost_at(self, step_cost: float) -> float:
        return self.cost * step_cost + self.fixed_cost


@dataclass(frozen=True)
class Job:
    """Step types (ints), token costs and irreversibility flags for steps 1..n (stored 0-based)."""

    types: np.ndarray
    costs: np.ndarray
    irreversible: np.ndarray
    name: str = ""

    def __post_init__(self):
        object.__setattr__(self, "types", np.asarray(self.types, dtype=int))
        object.__setattr__(self, "costs", np.asarray(self.costs, dtype=float))
        object.__setattr__(self, "irreversible", np.asarray(self.irreversible, dtype=bool))
        if not (len(self.types) == len(self.costs) == len(self.irreversible)):
            raise ValueError("types, costs and irreversible must have one entry per step")

    @property
    def n(self) -> int:
        return len(self.types)

    def last_commit_before(self) -> np.ndarray:
        """L[b] for b = 0..n: the last irreversible step strictly before b (0 if none)."""
        out = np.zeros(self.n + 1, dtype=int)
        last = 0
        for b in range(1, self.n + 1):
            out[b] = last
            if self.irreversible[b - 1]:
                last = b
        return out


@dataclass(frozen=True)
class RunModel:
    """Hazard per step type, the check menu, frailty and recovery parameters.

    retry_fresh / retry_inplace multiply the hazard of a retry that rolls back to the last passed
    check with a fresh context (and the check's reason), or that retries with the failed attempt still
    in context. rollback_cost is the extra tokens of a fresh retry, as a multiple of the step cost at
    the check (re-reading a summary of the history).
    """

    hazard: tuple[float, ...]
    checks: tuple[Check, ...] = field(default_factory=tuple)
    frailty_var: float = 0.0
    retry_fresh: float = 1.0
    retry_inplace: float = 1.0
    max_retries: int = 2
    rollback_cost: float = 0.35

    def eta(self, job: Job) -> np.ndarray:
        return np.asarray(self.hazard, dtype=float)[job.types]

    def retry_multiplier(self, mode: str) -> float:
        return self.retry_fresh if mode == "fresh" else self.retry_inplace


def frailty_nodes(var: float, q: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Equal-weight quadrature for Z ~ Gamma(mean 1, variance var): quantile midpoints."""
    if var <= 0:
        return np.array([1.0]), np.array([1.0])
    shape = 1.0 / var
    u = (np.arange(q) + 0.5) / q
    z = stats.gamma.ppf(u, a=shape, scale=var)
    return z, np.full(q, 1.0 / q)


def _attempt(job: Job, model: RunModel, start: int, b: int, check: Check, mu: float) -> tuple[float, float]:
    """On-track probability and probability of a caught error for steps start+1..b at hazard multiplier mu."""
    eta = model.eta(job)
    surv = 1.0
    caught = 0.0
    for i in range(start + 1, b + 1):
        h = 1.0 - np.exp(-mu * eta[i - 1])
        caught += surv * h * check.covers(job.types[i - 1]) * check.recall(b - i)
        surv *= 1.0 - h
    return surv, caught


def segment_outcome(job: Job, model: RunModel, a: int, b: int, check: int, z: float = 1.0, mode: str = "fresh") -> tuple[float, float]:
    """Success probability and expected tokens of steps a+1..b followed by check ``check`` at b.

    Success means the run is on track at b once the segment's flags and retries are done. An error
    that the check misses, or that was made at or before a committed irreversible step, is fatal.
    """
    chk = model.checks[check]
    commit = max(a, int(job.last_commit_before()[b]))
    f = chk.false_flag
    cost_b = job.costs[b - 1]
    k = chk.cost_at(cost_b)

    # First attempt over a+1..b; only errors after the last commit are recoverable.
    p_pre, _ = _attempt(job, model, a, commit, chk, z)
    p_post, d_post = _attempt(job, model, commit, b, chk, z)
    p1, d1 = p_pre * p_post, p_pre * d_post
    ok1, flag1 = p1 * (1 - f), p1 * f + d1

    # Retries repeat commit+1..b.
    mu_r = z * model.retry_multiplier(mode)
    pr, dr = _attempt(job, model, commit, b, chk, mu_r)
    okr, flagr = pr * (1 - f), pr * f + dr

    c1 = job.costs[a:b].sum() + k
    cr = job.costs[commit:b].sum() + k + (model.rollback_cost * cost_b if mode == "fresh" else 0.0)
    R = model.max_retries
    if R == 0:
        return ok1 + p1 * f, c1
    geo = (1 - flagr**R) / (1 - flagr) if flagr < 1 else float(R)
    success = ok1 + flag1 * okr * geo + flag1 * flagr ** (R - 1) * pr * f
    cost = c1 + flag1 * cr * geo
    return float(success), float(cost)
