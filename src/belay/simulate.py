"""Monte Carlo simulation of long agent runs under a placement policy.

The simulated world can be richer than the model a policy was planned with: frailty can be lognormal
rather than gamma, the hazard can drift upwards over the run (context growth), a same-model critic can
share the agent's blind spots (its recall falls on hard tasks), and votes over samples of one step can
share the same mistake. The report measures every table on this simulator and asks how well the fitted
model, which has none of these effects, predicts it (Section 6.3).

A policy is a pair of tables next_b[d, a], next_v[d, a]: after a passed check at a, with d flags so far,
the next check is check next_v at step next_b (or no further check if next_v is -1). Static plans
ignore d; the adaptive policy (``belay.adaptive``) does not. Per-step voting is the one policy that is
not a placement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .model import Job, RunModel
from .placement import Plan


@dataclass(frozen=True)
class World:
    """The ground truth a run is simulated in."""

    model: RunModel
    frailty: str = "gamma"  # or "lognormal"
    frailty_var: float | None = None  # defaults to model.frailty_var
    drift: float = 0.0  # the last step's hazard is (1 + drift) times the first's, linearly in between
    blind_spot: tuple[float, ...] | None = None  # per check: recall scales with Z ** -blind_spot
    vote_corr: float = 0.0  # probability that the votes on a step share one outcome

    def sample_z(self, rng: np.random.Generator, size: int) -> np.ndarray:
        var = self.model.frailty_var if self.frailty_var is None else self.frailty_var
        if var <= 0:
            return np.ones(size)
        if self.frailty == "lognormal":
            s2 = np.log1p(var)
            return rng.lognormal(mean=-s2 / 2, sigma=np.sqrt(s2), size=size)
        return rng.gamma(shape=1 / var, scale=var, size=size)

    def eta(self, job: Job) -> np.ndarray:
        ramp = np.linspace(0.0, 1.0, job.n) if job.n > 1 else np.zeros(1)
        return self.model.eta(job) * (1 + self.drift * ramp)


@dataclass(frozen=True)
class Policy:
    name: str
    tables: Callable[[Job, RunModel], tuple[np.ndarray, np.ndarray]] | None = None
    mode: str = "fresh"
    votes: int = 0


@dataclass
class RunStats:
    success: np.ndarray
    tokens: np.ndarray
    checks: np.ndarray
    flags: np.ndarray
    retries: np.ndarray
    z: np.ndarray = field(repr=False)


def _first_applicable(job: Job, model: RunModel, v: int, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.array([model.checks[v].applies(t) for t in job.types])
    nb = np.full(job.n + 1, job.n, dtype=int)
    nv = np.full(job.n + 1, -1, dtype=int)
    for a in range(job.n):
        for b in positions[a]:
            if b <= job.n and ok[b - 1]:
                nb[a], nv[a] = b, v
                break
    return nb[None, :], nv[None, :]


def static(plan: Plan, mode: str = "fresh") -> Policy:
    return Policy("static", lambda job, model: (plan.next_b[None, :], plan.next_v[None, :]), mode)


def adaptive(plans: list[Plan], mode: str = "fresh") -> Policy:
    """Plans indexed by the number of flags so far; the last one is used for any larger number."""
    return Policy("adaptive", lambda job, model: (np.stack([p.next_b for p in plans]), np.stack([p.next_v for p in plans])), mode)


def every_step(check: int, mode: str = "inplace") -> Policy:
    return Policy(
        "every-step",
        lambda job, model: _first_applicable(job, model, check, [range(a + 1, job.n + 1) for a in range(job.n)]),
        mode,
    )


def periodic(check: int, every: int, mode: str = "fresh") -> Policy:
    def tables(job, model):
        return _first_applicable(job, model, check, [range((a // every + 1) * every, job.n + 1, every) for a in range(job.n)])

    return Policy(f"periodic-{every}", tables, mode)


def no_checks() -> Policy:
    return Policy("none", lambda job, model: (np.full((1, job.n + 1), job.n), np.full((1, job.n + 1), -1)))


def voting(k: int = 3) -> Policy:
    return Policy(f"vote-{k}", None, votes=k)


def simulate(job: Job, world: World, policy: Policy, runs: int, rng: np.random.Generator, z: np.ndarray | None = None) -> RunStats:
    """Simulate ``runs`` runs of ``job``; frailty draws can be passed in to pair runs across policies."""
    m = world.model
    n = job.n
    eta = world.eta(job)
    if z is None:
        z = world.sample_z(rng, runs)
    zeros = np.zeros(runs, dtype=int)

    if policy.votes:
        h = 1 - np.exp(-np.outer(z, eta))
        ok1 = 1 - h
        major = ok1**3 + 3 * ok1**2 * h if policy.votes == 3 else ok1  # majority of three
        step_ok = world.vote_corr * ok1 + (1 - world.vote_corr) * major
        success = rng.random(runs) < np.prod(step_ok, axis=1)
        tokens = np.full(runs, (policy.votes + 0.1) * job.costs.sum())
        return RunStats(success, tokens, zeros, zeros.copy(), zeros.copy(), z)

    NB, NV = policy.tables(job, m)
    D = NB.shape[0] - 1
    H = np.concatenate([[0.0], np.cumsum(eta)])
    Cc = np.concatenate([[0.0], np.cumsum(job.costs)])
    L = job.last_commit_before()
    V = len(m.checks)
    cov = np.array([[c.covers(t) for t in job.types] for c in m.checks]) if V else np.zeros((0, n))
    fflag = np.array([c.false_flag for c in m.checks] + [0.0])
    kcost = np.array([[c.cost_at(job.costs[b - 1]) for b in range(1, n + 1)] for c in m.checks]) if V else np.zeros((0, n))
    blind = np.zeros(V) if world.blind_spot is None else np.asarray(world.blind_spot, float)
    r0 = np.array([c.recall0 for c in m.checks] + [0.0])
    rinf = np.array([c.r_inf for c in m.checks] + [0.0])
    phi = np.array([c.decay for c in m.checks] + [0.0])
    rho = m.retry_multiplier(policy.mode)
    rollback = m.rollback_cost if policy.mode == "fresh" else 0.0

    a = np.zeros(runs, dtype=int)
    d = np.zeros(runs, dtype=int)
    doomed = np.zeros(runs, dtype=bool)
    err = np.zeros(runs, dtype=int)
    tokens = np.zeros(runs)
    tries = np.zeros(runs, dtype=int)
    segb = np.zeros(runs, dtype=int)
    segv = np.zeros(runs, dtype=int)
    start = np.zeros(runs, dtype=int)
    mult = np.ones(runs)
    done = np.zeros(runs, dtype=bool)
    nchecks = np.zeros(runs, dtype=int)
    nflags = np.zeros(runs, dtype=int)
    nretries = np.zeros(runs, dtype=int)

    while not done.all():
        act = np.flatnonzero(~done)
        fresh_seg = act[tries[act] == 0]
        dd = np.minimum(d[fresh_seg], D)
        segb[fresh_seg] = NB[dd, a[fresh_seg]]
        segv[fresh_seg] = NV[dd, a[fresh_seg]]
        start[fresh_seg] = a[fresh_seg]
        mult[fresh_seg] = 1.0

        b, v, s = segb[act], segv[act], start[act]
        target = H[s] + rng.exponential(size=len(act)) / (z[act] * mult[act])
        i = np.searchsorted(H, target, side="left")
        newerr = (~doomed[act]) & (i <= b)
        tokens[act] += Cc[b] - Cc[s]

        checked = v >= 0
        vc = np.where(checked, v, 0)
        tokens[act] += np.where(checked, kcost[vc, b - 1] if V else 0.0, 0.0)
        nchecks[act] += checked

        has_err = newerr | doomed[act]
        e = np.where(newerr, i, err[act])
        ec = np.clip(e, 1, n)
        lat = np.maximum(b - ec, 0)
        rec = rinf[vc] - (rinf[vc] - r0[vc]) * np.power(phi[vc], lat)
        p_err = cov[vc, ec - 1] * rec * np.power(z[act], -blind[vc]) if V else np.zeros(len(act))
        p_flag = np.where(has_err, np.clip(p_err, 0, 1), fflag[vc])
        flag = checked & (rng.random(len(act)) < p_flag)
        nflags[act] += flag
        d[act] += flag

        Lp = np.maximum(a[act], L[b])
        recoverable = newerr & (i > Lp)
        retry = flag & (tries[act] < m.max_retries)

        # Retry: roll back to the last passed check (or the last commit after it) and repeat.
        rr = act[retry]
        bad = (newerr & ~recoverable)[retry]
        doomed[rr[bad]] = True
        err[rr[bad]] = i[retry][bad]
        tokens[rr] += rollback * job.costs[b[retry] - 1]
        tries[rr] += 1
        nretries[rr] += 1
        start[rr] = Lp[retry]
        mult[rr] = rho

        # Proceed: whatever was not caught stays.
        go = act[~retry]
        miss = newerr[~retry]
        doomed[go[miss]] = True
        err[go[miss]] = i[~retry][miss]
        a[go] = b[~retry]
        tries[go] = 0
        done[go] = a[go] >= n

    return RunStats(~doomed, tokens, nchecks, nflags, nretries, z)
