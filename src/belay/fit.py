"""Fitting the derailment model from labelled development traces (Section 4.3 of the report).

Development runs execute with no intervention while every applicable check runs in shadow mode after
every step; each run is labelled with its first derailing step. At a sample of caught errors the
derailed step is retried once in each recovery mode. From these traces:

* per-type hazards and the gamma frailty variance maximise the marginal likelihood of the first
  derailment times, with one frailty draw shared by all runs of a task (integrated by quadrature);
* each check's false-flag rate is its flag rate on on-track steps, and its recall is fitted as
  a_k * (rho_inf + (1 - rho_inf) * decay ** d) for an error of type k made d steps earlier, which gives
  recall0 = max_k a_k, coverage_k = a_k / recall0 and recall_inf = rho_inf * recall0;
* each recovery mode's hazard multiplier maximises the likelihood of the retry outcomes, integrating
  the task's frailty over its posterior given when the run derailed. This separates contamination
  (rho > 1) from the selection effect (the retry is drawn from harder tasks).

``frailty=False, latency=False, retry_modes=False`` gives the independent-step fit the report uses
as the baseline model: no frailty, recall pooled over latency, retries as fresh draws.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import optimize
from scipy.special import expit

from .model import Job, RunModel, frailty_nodes
from .simulate import World


@dataclass
class Traces:
    run_job: np.ndarray  # job index of each run
    first_error: np.ndarray  # first derailing step (1-based), 0 if the run never derailed
    obs_run: np.ndarray  # check observations: run, check, error type (-1 on track), latency, flagged
    obs_check: np.ndarray
    obs_type: np.ndarray
    obs_latency: np.ndarray
    obs_flag: np.ndarray
    retry_run: np.ndarray  # retry observations: run, mode (0 fresh, 1 in place), succeeded
    retry_mode: np.ndarray
    retry_ok: np.ndarray


def shadow_traces(jobs: list[Job], world: World, runs_per_job: int, rng: np.random.Generator, z: np.ndarray | None = None) -> Traces:
    """Labelled development traces of ``runs_per_job`` runs of each job in ``world``.

    A job is one task: its runs share the task's frailty, drawn from the world unless ``z`` gives one
    value per job.
    """
    m = world.model
    cols = {k: [] for k in ("run_job", "first_error", "obs_run", "obs_check", "obs_type", "obs_latency", "obs_flag", "retry_run", "retry_mode", "retry_ok")}
    blind = np.zeros(len(m.checks)) if world.blind_spot is None else np.asarray(world.blind_spot, float)
    z_tasks = z
    base = 0
    for j, job in enumerate(jobs):
        R, n = runs_per_job, job.n
        z = np.full(R, world.sample_z(rng, 1)[0] if z_tasks is None else z_tasks[j])
        eta = world.eta(job)
        H = np.concatenate([[0.0], np.cumsum(eta)])
        T = np.searchsorted(H, rng.exponential(size=R) / z, side="left")
        T = np.where(T <= n, T, 0)
        cols["run_job"].append(np.full(R, j))
        cols["first_error"].append(T)
        runs = base + np.arange(R)
        for v, chk in enumerate(m.checks):
            pos = np.array([b for b in range(1, n + 1) if chk.applies(job.types[b - 1])], dtype=int)
            if not len(pos):
                continue
            rr, bb = np.meshgrid(np.arange(R), pos, indexing="ij")
            rr, bb = rr.ravel(), bb.ravel()
            tt = T[rr]
            on = (tt == 0) | (bb < tt)
            et = np.where(on, -1, job.types[np.maximum(tt, 1) - 1])
            lat = np.where(on, -1, bb - tt)
            cov = np.array([chk.covers(k) for k in range(len(m.hazard))])
            p_err = cov[np.maximum(et, 0)] * chk.recall(np.maximum(lat, 0)) * np.minimum(1.0, np.power(z[rr], -blind[v]))
            p = np.where(on, chk.false_flag, np.clip(p_err, 0, 1))
            cols["obs_run"].append(runs[rr])
            cols["obs_check"].append(np.full(len(rr), v))
            cols["obs_type"].append(et)
            cols["obs_latency"].append(lat)
            cols["obs_flag"].append(rng.random(len(rr)) < p)
        bad = np.flatnonzero(T > 0)
        for mode, rho in ((0, m.retry_fresh), (1, m.retry_inplace)):
            ok = rng.random(len(bad)) < np.exp(-rho * z[bad] * eta[T[bad] - 1])
            cols["retry_run"].append(runs[bad])
            cols["retry_mode"].append(np.full(len(bad), mode))
            cols["retry_ok"].append(ok)
        base += R
    return Traces(**{k: np.concatenate(v) for k, v in cols.items()})


def _counts(jobs: list[Job], tr: Traces, n_types: int):
    """Per run: type counts of steps before the first error, the error's type, and all steps."""
    N = len(tr.run_job)
    before = np.zeros((N, n_types))
    total = np.zeros((N, n_types))
    etype = np.full(N, -1)
    for r in range(N):
        job = jobs[tr.run_job[r]]
        total[r] = np.bincount(job.types, minlength=n_types)
        T = tr.first_error[r]
        if T > 0:
            before[r] = np.bincount(job.types[: T - 1], minlength=n_types)
            etype[r] = job.types[T - 1]
    return before, total, etype


def _run_loglik(before, total, etype, eta, z):
    """log P(first error at T | z) for every run and frailty node: shape (runs, Q)."""
    event = etype >= 0
    Hb = before @ eta
    Ht = Hb + np.where(event, eta[np.maximum(etype, 0)], 0.0)
    Hn = total @ eta
    ev = -np.outer(Hb, z) + np.log(-np.expm1(-np.outer(Ht - Hb, z).clip(1e-12)))
    return np.where(event[:, None], ev, -np.outer(Hn, z))


def _task_posterior(tr: Traces, before, total, etype, eta, var, q: int = 48):
    """Frailty nodes and, per task, posterior log-weights given all its runs: shape (tasks, Q)."""
    z, w = frailty_nodes(var, q)
    ll = _run_loglik(before, total, etype, eta, z)
    tasks = np.unique(tr.run_job)
    per_task = np.zeros((tasks.max() + 1, len(z)))
    np.add.at(per_task, tr.run_job, ll)
    return z, np.log(w)[None, :] + per_task


def _fit_hazards(tr: Traces, before, total, etype, frailty: bool, n_types: int):
    """Marginal likelihood of first-error times with one gamma frailty draw per task."""

    def nll(theta):
        eta = np.exp(theta[:n_types])
        var = float(np.exp(theta[n_types])) if frailty else 0.0
        _, lp = _task_posterior(tr, before, total, etype, eta, var)
        top = lp.max(axis=1, keepdims=True)
        return -(top[:, 0] + np.log(np.exp(lp - top).sum(axis=1))).sum()

    x0 = np.concatenate([np.full(n_types, np.log(0.03)), [np.log(0.5)]])
    res = optimize.minimize(nll, x0, method="L-BFGS-B")
    eta = np.exp(res.x[:n_types])
    var = float(np.exp(res.x[n_types])) if frailty else 0.0
    return eta, var


def _fit_check(chk, tr: Traces, v: int, n_types: int, latency: bool):
    sel = tr.obs_check == v
    on = sel & (tr.obs_type < 0)
    f = float(tr.obs_flag[on].mean()) if on.any() else chk.false_flag
    err = sel & (tr.obs_type >= 0)
    k, d, y = tr.obs_type[err], tr.obs_latency[err].astype(float), tr.obs_flag[err].astype(float)
    seen = np.bincount(k, minlength=n_types) > 0

    def unpack(theta):
        a = expit(theta[:n_types])
        rho = np.exp(theta[n_types]) if latency else 1.0
        phi = expit(theta[n_types + 1]) if latency else 0.5
        return a, rho, phi

    def nll(theta):
        a, rho, phi = unpack(theta)
        p = np.clip(a[k] * (rho + (1 - rho) * phi**d), 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log1p(-p)).sum() + 1e-3 * (theta**2).sum()

    res = optimize.minimize(nll, np.zeros(n_types + 2), method="L-BFGS-B")
    a, rho, phi = unpack(res.x)
    prior = np.array([chk.covers(t) * chk.recall0 for t in range(n_types)])
    a = np.where(seen, a, prior)
    r0 = float(a.max()) if a.max() > 0 else chk.recall0
    return replace(
        chk,
        recall0=r0,
        recall_inf=float(min(1.0, rho * r0)),
        decay=float(phi),
        false_flag=f,
        coverage=tuple(float(x) for x in a / r0),
    )


def _fit_retry(tr: Traces, eta, var, before, total, etype, mode: int) -> float:
    sel = tr.retry_mode == mode
    runs, ok = tr.retry_run[sel], tr.retry_ok[sel].astype(float)
    z, lp = _task_posterior(tr, before, total, etype, eta, var)
    post = np.exp(lp - lp.max(axis=1, keepdims=True))
    post = (post / post.sum(axis=1, keepdims=True))[tr.run_job[runs]]
    et = eta[etype[runs]]

    def nll(log_rho):
        p = (post * np.exp(-np.exp(log_rho) * np.outer(et, z))).sum(axis=1)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(ok * np.log(p) + (1 - ok) * np.log1p(-p)).sum()

    res = optimize.minimize_scalar(nll, bounds=(np.log(0.05), np.log(8.0)), method="bounded")
    return float(np.exp(res.x))


def fit(jobs: list[Job], traces: Traces, template: RunModel, *, frailty: bool = True, latency: bool = True, retry_modes: bool = True) -> RunModel:
    """A RunModel fitted to ``traces``; costs, check menu and retry cap come from ``template``."""
    n_types = len(template.hazard)
    before, total, etype = _counts(jobs, traces, n_types)
    eta, var = _fit_hazards(traces, before, total, etype, frailty, n_types)
    checks = tuple(_fit_check(c, traces, v, n_types, latency) for v, c in enumerate(template.checks))
    if retry_modes:
        fresh = _fit_retry(traces, eta, var, before, total, etype, 0)
        inplace = _fit_retry(traces, eta, var, before, total, etype, 1)
    else:
        fresh = inplace = 1.0
    return replace(template, hazard=tuple(float(x) for x in eta), checks=checks, frailty_var=var, retry_fresh=fresh, retry_inplace=inplace)
