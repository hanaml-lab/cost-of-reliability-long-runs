"""Seeded generator for every number in the report (illustrative, simulated results).

Writes results.json and tables/*.tex; figures.py reads results.json only. Every number is *computed*:
the four environments are simulated with ``belay.simulate`` in a ground-truth world that is richer
than the planning model (lognormal task frailty, hazard drift over the run, a same-model critic that
shares the agent's blind spots, correlated votes); the planning models are *fitted* with ``belay.fit``
on cross-fitted development traces; and every policy is evaluated on the same tasks and frailty draws.
Tables, text and figures therefore agree by construction.

Requires the ``belay`` package (pip install -e .. from the repository root, or BELAY_SRC=<repo>/src).
Runs in about ten minutes on 16 CPU cores.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("BELAY_SRC", os.path.join(ROOT, "..", "src")))

from belay.adaptive import posterior_mean  # noqa: E402
from belay.fit import fit, shadow_traces  # noqa: E402
from belay.model import Check, Job, RunModel  # noqa: E402
from belay.placement import segment_tables, _dp  # noqa: E402
from belay.simulate import World, adaptive, every_step, no_checks, periodic, simulate, static, voting  # noqa: E402
from belay.theory import retry_success  # noqa: E402

os.makedirs(os.path.join(ROOT, "tables"), exist_ok=True)
MASTER = 20251209
SEEDS = [0, 1, 2, 3, 4] if not os.environ.get("MINI") else [0, 1]
FOLDS = 5
DEV_RUNS = 8  # shadow-mode development runs per task
LAMS = [0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3]  # success per unchecked run of tokens
LAM_STAR = 0.2
MAIN = "static"  # the headline method: the plan solved once per job; "belay" is the variant that re-plans after flags
PLACE_LAMS = (0.005, 0.03, 0.1)
PERIODS = [2, 3, 4, 6, 8, 12, 16]
Q = 16
DMAX = 3
PRED_DRAWS = 24  # frailty draws per task when a fitted model predicts a policy
WORKERS = int(os.environ.get("WORKERS", "16"))


# ---------------------------------------------------------------- environments

def ramp_costs(n, scale=1.0):
    return scale * (1 + 0.6 * np.arange(n) / max(n - 1, 1))


def gen_retail(rng, airline=False):
    seq = [3]
    seq += [0] * int(rng.integers(1, 4))
    for _ in range(int(rng.integers(1, 4))):
        seq += [0] * int(rng.integers(0, 3)) + [1] * int(rng.integers(1, 3 if not airline else 4)) + [3, 2]
        if rng.random() < 0.5:
            seq += [3]
    seq += [3]
    return seq


def gen_tac(rng):
    n = int(np.clip(rng.lognormal(np.log(40), 0.4), 15, 90))
    seq = []
    while len(seq) < n:
        r = rng.random()
        if r < 0.32:
            seq += [0]
        elif r < 0.55:
            seq += [1]
        elif r < 0.75:
            seq += [2]
        elif r < 0.88:
            seq += [3]
        else:
            seq += [4]
    return seq[:n]


def gen_backoffice(rng):
    target = int(np.clip(rng.lognormal(np.log(52), 0.5), 18, 140))
    seq = []
    while len(seq) < target:
        seq += [0] * int(rng.integers(1, 3)) + [1]
        if rng.random() < 0.3:
            seq += [2]
        if rng.random() < 0.2:
            seq += [3]
        seq += [4]
        if rng.random() < 0.5:
            seq += [5]
    return seq[:target]


ENVS = {
    "retail": dict(
        title="τ-bench retail", tasks=115, trials=8, gen=lambda r: gen_retail(r), irrev=(2,),
        types=["lookup", "decision", "write", "message"],
        eta=[0.020, 0.075, 0.040, 0.012], target=0.522, pass4=0.26,
        checks=[
            Check("schema", cost=0.0, fixed_cost=0.03, recall0=0.97, recall_inf=0.0, decay=0.0, false_flag=0.004, coverage=(0.55, 0.0, 0.45, 0.0), after=(0, 2)),
            Check("policy-rule", cost=0.0, fixed_cost=0.05, recall0=0.93, recall_inf=0.93, false_flag=0.01, coverage=(0.15, 0.55, 0.50, 0.0), after=(2,)),
            Check("critic", cost=0.80, recall0=0.64, recall_inf=0.26, decay=0.55, false_flag=0.07),
        ],
        blind=(0.0, 0.0, 0.35), critic=2,
    ),
    "airline": dict(
        title="τ-bench airline", tasks=50, trials=8, gen=lambda r: gen_retail(r, airline=True), irrev=(2,),
        types=["lookup", "decision", "write", "message"],
        eta=[0.022, 0.110, 0.045, 0.012], target=0.290, pass4=0.12,
        checks=[
            Check("schema", cost=0.0, fixed_cost=0.03, recall0=0.97, recall_inf=0.0, decay=0.0, false_flag=0.004, coverage=(0.55, 0.0, 0.45, 0.0), after=(0, 2)),
            Check("policy-rule", cost=0.0, fixed_cost=0.05, recall0=0.90, recall_inf=0.90, false_flag=0.015, coverage=(0.15, 0.45, 0.45, 0.0), after=(2,)),
            Check("critic", cost=0.80, recall0=0.60, recall_inf=0.24, decay=0.55, false_flag=0.08),
        ],
        blind=(0.0, 0.0, 0.35), critic=2,
    ),
    "tac": dict(
        title="TheAgentCompany", tasks=175, trials=4, gen=gen_tac, irrev=(3, 4),
        types=["navigate", "edit", "reason", "message", "submit"],
        eta=[0.018, 0.050, 0.040, 0.030, 0.035], target=0.092, pass4=None,
        checks=[
            Check("artifact", cost=0.0, fixed_cost=0.05, recall0=0.92, recall_inf=0.60, decay=0.5, false_flag=0.01, coverage=(0.0, 0.6, 0.3, 0.0, 0.5), after=(1, 4)),
            Check("state-probe", cost=0.15, recall0=0.30, recall_inf=0.75, decay=0.8, false_flag=0.03, coverage=(0.6, 0.4, 0.3, 0.5, 0.7)),
            Check("critic", cost=0.80, recall0=0.58, recall_inf=0.24, decay=0.55, false_flag=0.08),
        ],
        blind=(0.0, 0.0, 0.35), critic=2,
    ),
    "backoffice": dict(
        title="Back-office jobs", tasks=200, trials=4, gen=gen_backoffice, irrev=(3, 4, 5),
        types=["read", "match", "resolve", "approval", "pay", "notify"],
        eta=[0.004, 0.035, 0.070, 0.010, 0.015, 0.003], target=0.49, pass4=None,
        checks=[
            Check("record-exists", cost=0.0, fixed_cost=0.02, recall0=0.96, recall_inf=0.0, decay=0.0, false_flag=0.003, coverage=(0.8, 0.0, 0.0, 0.0, 0.0, 0.0), after=(0,)),
            Check("totals", cost=0.0, fixed_cost=0.03, recall0=0.97, recall_inf=0.97, false_flag=0.005, coverage=(0.0, 0.7, 0.35, 0.0, 0.6, 0.0), after=(1, 4)),
            Check("critic", cost=0.80, recall0=0.62, recall_inf=0.27, decay=0.55, false_flag=0.075),
        ],
        blind=(0.0, 0.0, 0.35), critic=2,
    ),
}
ENV_ORDER = ["retail", "airline", "tac", "backoffice"]
RETRY = dict(retry_fresh=0.85, retry_inplace=1.9, max_retries=2, rollback_cost=0.35)
DRIFT = 0.25
VOTE_CORR = 0.55


def build_jobs(name):
    env = ENVS[name]
    rng = np.random.default_rng([MASTER, ENV_ORDER.index(name)])
    jobs = []
    for t in range(env["tasks"]):
        seq = np.array(env["gen"](rng), dtype=int)
        jobs.append(Job(types=seq, costs=ramp_costs(len(seq)), irreversible=np.isin(seq, env["irrev"]), name=f"{name}-{t}"))
    return jobs


def lognormal_z(rng, var, size):
    s2 = math.log1p(var)
    return rng.lognormal(-s2 / 2, math.sqrt(s2), size)


def calibrate(name, jobs):
    """Task frailties and a hazard scale that reproduce the single agent's pass^1 and pass^4.

    Frailty is gamma with mean one; its variance is set so that the single agent's pass^4 matches the
    target (tasks a model almost never solves make pass^k fall slowly), then the hazard scale so that
    its pass^1 does. The draws are fixed per task: a benchmark's tasks do not change between seeds.
    Returns the task frailties, the hazard scale and the frailty variance.
    """
    env = ENVS[name]
    eta = np.array(env["eta"])
    H = []
    for j in jobs:
        ramp = np.linspace(0, 1, j.n) if j.n > 1 else np.zeros(1)
        H.append((eta[j.types] * (1 + DRIFT * ramp)).sum())
    H = np.array(H)
    u = (np.random.default_rng([MASTER, 99, ENV_ORDER.index(name)]).permutation(len(jobs)) + 0.5) / len(jobs)

    def at(var):
        z = stats.gamma.ppf(u, a=1 / var, scale=var)
        z = z / z.mean()
        lo, hi = 1e-3, 100.0
        for _ in range(80):
            mid = math.sqrt(lo * hi)
            if np.exp(-z * H * mid).mean() > env["target"]:
                lo = mid
            else:
                hi = mid
        return z, mid, np.exp(-z * H * mid)

    if env["pass4"] is None:
        var = 1.5
    else:
        lo, hi = 0.1, 20.0
        for _ in range(50):
            var = math.sqrt(lo * hi)
            _, _, p = at(var)
            if (p**4).mean() < env["pass4"]:
                lo = var
            else:
                hi = var
    z, scale, _ = at(var)
    return z, scale, var


def make_world(name, scale, var):
    env = ENVS[name]
    checks = tuple(env["checks"])
    blind = tuple(env["blind"])
    model = RunModel(hazard=tuple(float(x) for x in np.array(env["eta"]) * scale), checks=checks, frailty_var=var, **RETRY)
    return World(model, frailty="gamma", frailty_var=var, drift=DRIFT, blind_spot=blind, vote_corr=VOTE_CORR)


# ---------------------------------------------------------------- evaluation helpers

def pass_hat_k(successes, trials, k):
    """Unbiased pass^k from c successes in n trials per task (tau-bench): C(c,k)/C(n,k)."""
    return float(np.mean([math.comb(int(c), k) / math.comb(trials, k) for c in successes]))


def priced(job, tab, lam0, floor=0.02):
    """The plan for one job at price lam0 on the linear objective E[success] - lam0 * E[tokens] / C0.

    log S - (lam / S*) C has the same first-order condition as S - lam C at S = S*, so the log-additive
    program is solved at the local price lam0 / (C0 * S*), with S* first the unchecked success and then
    the success of the resulting plan (one fixed-point step).
    """
    c0 = job.costs.sum()
    p = _dp(job, tab, lam0 / (c0 * max(float(tab[2][0]), floor)))
    return _dp(job, tab, lam0 / (c0 * max(p.success, floor)))


def local_price(job, tab, lam0, floor=0.02):
    """The log-program price lam0 / (C0 * S*) that matches the linear objective at the plan's success S*."""
    c0 = job.costs.sum()
    p = _dp(job, tab, lam0 / (c0 * max(float(tab[2][0]), floor)))
    return lam0 / (c0 * max(p.success, floor))


def plans_for(job, model, lams, detections_max, q):
    """Plans per flag count and price. The price is set once per job, from the unflagged plan, so that a
    flag changes the plan through the frailty posterior and not through a new price."""
    tabs = [segment_tables(job, model, q=q, detections=d) for d in range(detections_max + 1)]
    prices = [local_price(job, tabs[0], lam) for lam in lams]
    return [[_dp(job, tab, pr) for pr in prices] for tab in tabs]  # [d][lambda index]


def accuracy_model(model, base_rate):
    """The planning model with each check's recall replaced by its accuracy (the mis-ranking ablation)."""
    checks = []
    for c in model.checks:
        acc = (1 - base_rate) * (1 - c.false_flag) + base_rate * c.recall0
        checks.append(replace(c, recall0=acc, recall_inf=acc, coverage=None))
    return replace(model, checks=tuple(checks))


def run_env_seed(args):
    """All policies on one environment for one seed, with cross-fitted planning models."""
    name, seed = args
    env = ENVS[name]
    jobs = build_jobs(name)
    z_true, scale, var = calibrate(name, jobs)
    world = make_world(name, scale, var)
    rng = np.random.default_rng([MASTER, ENV_ORDER.index(name), seed])
    T = env["trials"]
    nt = len(jobs)
    folds = np.array_split(rng.permutation(nt), FOLDS)
    crit = env["critic"]
    base_cost = np.array([j.costs.sum() for j in jobs])

    res = {}  # policy -> dict(success=[nt x T], tokens=[nt x T])

    def record(key, t, out):
        r = res.setdefault(key, dict(success=np.zeros((nt, T), bool), tokens=np.zeros((nt, T))))
        r["success"][t] = out.success
        r["tokens"][t] = out.tokens

    preds = {}  # (model kind, policy) -> predicted success per task
    fitted_params = []
    traces_store = []
    for fi, test in enumerate(folds):
        dev = np.setdiff1d(np.arange(nt), test)
        dev_jobs = [jobs[i] for i in dev]
        tr = shadow_traces(dev_jobs, world, DEV_RUNS, rng, z=z_true[dev])
        traces_store.append((dev, tr))
        m_belay = fit(dev_jobs, tr, template=world.model)
        m_indep = fit(dev_jobs, tr, template=world.model, frailty=False, latency=False, retry_modes=False)
        m_nofrail = fit(dev_jobs, tr, template=world.model, frailty=False)
        m_nolat = fit(dev_jobs, tr, template=world.model, latency=False)
        base_rate = float((tr.first_error > 0).mean()) / max(np.mean([j.n for j in dev_jobs]), 1)
        m_acc = accuracy_model(m_belay, base_rate)
        fitted_params.append(dict(
            hazard=list(m_belay.hazard), frailty_var=m_belay.frailty_var, retry_fresh=m_belay.retry_fresh, retry_inplace=m_belay.retry_inplace,
            checks=[dict(name=c.name, recall0=c.recall0, recall_inf=c.r_inf, decay=c.decay, false_flag=c.false_flag, coverage=list(c.coverage)) for c in m_belay.checks],
            indep_hazard=list(m_indep.hazard), indep_recall=[c.recall0 for c in m_indep.checks],
        ))
        for t in test:
            job = jobs[t]
            star = LAMS.index(LAM_STAR)
            zt = np.full(T, z_true[t])
            P = plans_for(job, m_belay, LAMS, DMAX, Q)
            Pi = plans_for(job, m_indep, LAMS, 0, 1)[0]

            def sim(policy, key):
                out = simulate(job, world, policy, T, rng, z=zt)
                record(key, t, out)

            sim(no_checks(), "none")
            sim(voting(3), "vote3")
            sim(every_step(crit, mode="inplace"), "every")
            sim(every_step(crit, mode="fresh"), "every-fresh")
            for T_ in PERIODS:
                sim(periodic(crit, T_), f"periodic-{T_}")
            for li in range(len(LAMS)):
                sim(adaptive([P[d][li] for d in range(DMAX + 1)]), f"belay-{li}")
                sim(static(P[0][li]), f"static-{li}")
                sim(static(Pi[li]), f"indep-{li}")
            # ablations at the default price
            lam = LAMS[star]
            # ablations of the (static) plan at the default price
            abl = {
                "no-frailty": plans_for(job, m_nofrail, [lam], 0, 1)[0][0],
                "no-latency": plans_for(job, m_nolat, [lam], 0, Q)[0][0],
                "accuracy": plans_for(job, m_acc, [lam], 0, Q)[0][0],
            }
            for k, plan in abl.items():
                sim(static(plan), f"abl-{k}")
            m_inpl = replace(m_belay, retry_fresh=m_belay.retry_inplace, rollback_cost=0.0)
            sim(static(plans_for(job, m_inpl, [lam], 0, Q)[0][0], mode="inplace"), "abl-inplace")
            job_ni = Job(types=job.types, costs=job.costs, irreversible=np.zeros(job.n, bool))
            sim(static(plans_for(job_ni, m_belay, [lam], 0, Q)[0][0]), "abl-no-irreversible")

            # predictions of static policies by the two fitted models
            prng = np.random.default_rng([MASTER, 7, ENV_ORDER.index(name), seed, int(t)])
            pol = {"none": no_checks(), "every": every_step(crit, mode="inplace"), "periodic-3": periodic(crit, 3), "periodic-6": periodic(crit, 6), "periodic-12": periodic(crit, 12)}
            for li in (2, 5, 7, 10):
                pol[f"static-{li}"] = static(P[0][li])
                pol[f"indep-{li}"] = static(Pi[li])
            for kind, mdl in (("belay", m_belay), ("indep", m_indep)):
                w = World(mdl, frailty="gamma")
                zz = np.repeat(w.sample_z(prng, PRED_DRAWS), T)
                for key, policy in pol.items():
                    out = simulate(job, w, policy, len(zz), prng, z=zz)
                    pk = out.success.reshape(PRED_DRAWS, T)
                    preds.setdefault((kind, key), np.zeros((nt, 2)))
                    k4 = min(4, T)
                    preds[(kind, key)][t] = [pk.mean(), np.mean([math.comb(int(r.sum()), k4) / math.comb(T, k4) for r in pk])]

    out = {"policies": {}, "pred": {}, "fits": fitted_params, "scale": scale, "var": var}
    for key, r in res.items():
        succ_task = r["success"].sum(axis=1)
        out["policies"][key] = dict(
            success=float(r["success"].mean()),
            tokens=float(r["tokens"].mean() / base_cost.mean()),
            pass_k=[pass_hat_k(succ_task, T, k) for k in range(1, T + 1)],
            per_task=r["success"].mean(axis=1).tolist(),
        )
    for (kind, key), arr in preds.items():
        out["pred"][f"{kind}:{key}"] = dict(success=float(arr[:, 0].mean()), pass4=float(arr[:, 1].mean()))

    # retry decomposition from the development traces of the first fold, under the first fold's fit
    if seed == 0:
        out["retry"] = retry_decomposition(name, jobs, folds, traces_store[0], fitted_params[0], z_true)
    return name, seed, out


def retry_decomposition(name, jobs, folds, store, fp, z_true):
    """Measured first-attempt and retry success by step type, and the selection-only prediction."""
    dev, tr = store
    env = ENVS[name]
    types = env["types"]
    rows = []
    for k in range(len(types)):
        sel = tr.first_error > 0
        runs = np.flatnonzero(sel)
        et = np.array([jobs[dev[tr.run_job[r]]].types[tr.first_error[r] - 1] for r in runs])
        mk = runs[et == k]
        if len(mk) < 20:
            continue
        # first-attempt success of this step type among all executions in dev runs
        att, ok = 0, 0
        for r in range(len(tr.run_job)):
            job = jobs[dev[tr.run_job[r]]]
            T = tr.first_error[r]
            upto = job.types[: (T if T > 0 else job.n)]
            att += int((upto == k).sum())
            ok += int((upto == k).sum()) - int(T > 0 and job.types[T - 1] == k)
        fresh = tr.retry_ok[(tr.retry_mode == 0) & np.isin(tr.retry_run, mk)].mean()
        inpl = tr.retry_ok[(tr.retry_mode == 1) & np.isin(tr.retry_run, mk)].mean()
        first, sel_only = retry_success(fp["hazard"][k], fp["frailty_var"], 1.0)
        rows.append(dict(type=types[k], first=ok / att, fresh=float(fresh), inplace=float(inpl), n=int(len(mk)), model_first=first, selection_only=sel_only))
    return rows


# ---------------------------------------------------------------- horizon study

SYN_CHECKS = (
    Check("rule", cost=0.0, fixed_cost=0.02, recall0=0.95, recall_inf=0.0, decay=0.0, false_flag=0.004, coverage=(0.5,)),
    Check("critic", cost=0.80, recall0=0.65, recall_inf=0.30, decay=0.55, false_flag=0.05),
)
HORIZONS = [10, 20, 50, 100, 200, 500, 1000] if not os.environ.get("MINI") else [10, 50, 200]
SYN_VARS = [0.0, 0.25, 0.5, 1.0, 2.0]
BUDGET = 1.5


def horizon_point(args):
    n, var, seed = args
    rng = np.random.default_rng([MASTER, 31, n, int(var * 100), seed])
    tasks = 120 if n <= 200 else 60
    job = Job(types=np.zeros(n, int), costs=ramp_costs(n), irreversible=np.zeros(n, bool))
    eta = 0.01
    truth = RunModel(hazard=(eta,), checks=SYN_CHECKS, frailty_var=var, **RETRY)
    world = World(truth, frailty="lognormal", frailty_var=var, drift=DRIFT, blind_spot=(0.0, 0.35), vote_corr=VOTE_CORR)
    plan_model = replace(truth, frailty_var=max(var, 1e-6) if var > 0 else 0.0)
    indep_model = replace(truth, frailty_var=0.0, retry_fresh=1.0, retry_inplace=1.0, checks=tuple(replace(c, recall_inf=c.recall0) for c in SYN_CHECKS))
    z = world.sample_z(rng, tasks)
    zz = np.repeat(z, 2)
    base = job.costs.sum()
    lams = [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.4]
    w = min(n, 64)
    res = {}

    def best_under_budget(policies):
        best = (0.0, None)
        for pol in policies:
            out = simulate(job, world, pol, len(zz), rng, z=zz)
            c = out.tokens.mean() / base
            if c <= BUDGET and out.success.mean() > best[0]:
                best = (float(out.success.mean()), float(c))
        return best

    res["none"] = (float(simulate(job, world, no_checks(), len(zz), rng, z=zz).success.mean()), 1.0)
    res["periodic"] = best_under_budget([periodic(1, T) for T in [2, 3, 4, 6, 8, 12, 16, 24, 32, 48]])
    tab_i = segment_tables(job, indep_model, q=1, window=w)
    res["indep"] = best_under_budget([static(_dp(job, tab_i, local_price(job, tab_i, l))) for l in lams])
    tabs = [segment_tables(job, plan_model, q=Q, detections=d, window=w) for d in range(DMAX + 1)]
    res["belay"] = best_under_budget([static(_dp(job, tabs[0], local_price(job, tabs[0], l))) for l in lams])
    res["belay-adaptive"] = best_under_budget([adaptive([_dp(job, tabs[d], local_price(job, tabs[0], l)) for d in range(DMAX + 1)]) for l in lams])
    nf_model = replace(truth, frailty_var=0.0)
    tab_nf = segment_tables(job, nf_model, q=1, window=w)
    res["belay-nf"] = best_under_budget([static(_dp(job, tab_nf, local_price(job, tab_nf, l))) for l in lams])
    ev = simulate(job, world, every_step(1, mode="inplace"), len(zz), rng, z=zz)
    res["every"] = (float(ev.success.mean()), float(ev.tokens.mean() / base))
    vt = simulate(job, world, voting(3), len(zz), rng, z=zz)
    res["vote3"] = (float(vt.success.mean()), float(vt.tokens.mean() / base))
    return n, var, seed, res


# ---------------------------------------------------------------- transfer study

def transfer_point(args):
    """Plans fitted on retail traces, deployed on a shifted target, with m target tasks for refitting.

    The target's tasks are split once per seed: up to 20 for refitting and the rest (at least 30) for
    evaluation, so every m is evaluated on the same tasks.
    """
    shift, n_target, seed = args
    rng = np.random.default_rng([MASTER, 41, ["airline", "stronger", "weaker"].index(shift), seed])
    src_jobs = build_jobs("retail")
    zs, scale_s, var_s = calibrate("retail", src_jobs)
    src_world = make_world("retail", scale_s, var_s)
    if shift == "airline":
        tgt_jobs = build_jobs("airline")
        zt, scale_t, var_t = calibrate("airline", tgt_jobs)
        tgt_world = make_world("airline", scale_t, var_t)
    else:
        tgt_jobs = src_jobs
        zt = zs
        factor = 0.55 if shift == "stronger" else 1.6
        crit = ENVS["retail"]["checks"][2]
        crit_t = replace(crit, recall0=crit.recall0 * (0.85 if shift == "stronger" else 1.05), false_flag=crit.false_flag * (1.4 if shift == "stronger" else 0.9))
        m = replace(src_world.model, hazard=tuple(h * factor for h in src_world.model.hazard), checks=tuple(src_world.model.checks[:2]) + (crit_t,))
        tgt_world = replace(src_world, model=m)
    perm = rng.permutation(len(tgt_jobs))
    pool_dev, test = perm[:20], perm[20:]
    src_tr = shadow_traces(src_jobs, src_world, DEV_RUNS, rng, z=zs)
    m_src = fit(src_jobs, src_tr, template=src_world.model)
    if n_target > 0:
        dev = pool_dev[:n_target]
        tr = shadow_traces([tgt_jobs[i] for i in dev], tgt_world, DEV_RUNS, rng, z=zt[dev])
        # pooled refit: the target traces with the source traces down-weighted to 20 tasks' worth
        keep = rng.choice(len(src_jobs), size=20, replace=False)
        sub = shadow_traces([src_jobs[i] for i in keep], src_world, DEV_RUNS, rng, z=zs[keep])
        mdl = fit([src_jobs[i] for i in keep] + [tgt_jobs[i] for i in dev], merge_traces(sub, tr, len(keep)), template=tgt_world.model)
    else:
        mdl = m_src
    star = LAMS.index(LAM_STAR)
    s, c = [], []
    erng = np.random.default_rng([MASTER, 42, ["airline", "stronger", "weaker"].index(shift), seed])  # same trials for every m
    for t in test:
        job = tgt_jobs[t]
        plan = plans_for(job, mdl, [LAMS[star]], 0, Q)[0][0]
        out = simulate(job, tgt_world, static(plan), 8, erng, z=np.full(8, zt[t]))
        s.append(out.success.mean())
        c.append(out.tokens.mean() / job.costs.sum())
    return shift, n_target, seed, float(np.mean(s)), float(np.mean(c))


TRANSFER_GRID = [(sh, n, s) for sh in ("airline", "stronger", "weaker") for n in (0, 5, 10, 20) for s in SEEDS]


def merge_traces(a, b, offset_jobs):
    from belay.fit import Traces
    na = len(a.run_job)
    return Traces(
        run_job=np.concatenate([a.run_job, b.run_job + offset_jobs]),
        first_error=np.concatenate([a.first_error, b.first_error]),
        obs_run=np.concatenate([a.obs_run, b.obs_run + na]),
        obs_check=np.concatenate([a.obs_check, b.obs_check]),
        obs_type=np.concatenate([a.obs_type, b.obs_type]),
        obs_latency=np.concatenate([a.obs_latency, b.obs_latency]),
        obs_flag=np.concatenate([a.obs_flag, b.obs_flag]),
        retry_run=np.concatenate([a.retry_run, b.retry_run + na]),
        retry_mode=np.concatenate([a.retry_mode, b.retry_mode]),
        retry_ok=np.concatenate([a.retry_ok, b.retry_ok]),
    )


# ---------------------------------------------------------------- durability (crash injection)

DUR = dict(notice=0.9973, diverge=0.095, inflight=0.035, nondet=0.0005, snapshot=10, journal_ms=72.0, step_ms=2100.0)


def durability(seed):
    """One injected worker crash per back-office job, under four recovery designs."""
    rng = np.random.default_rng([MASTER, 51, seed])
    jobs = build_jobs("backoffice")
    z, scale, var = calibrate("backoffice", jobs)
    world = make_world("backoffice", scale, var)
    out = {k: dict(retained=0, base=0, dups=0) for k in ("scratch", "snapshot", "durable", "durable-keys")}
    over = []
    for t, job in enumerate(jobs):
        lam = LAM_STAR / job.costs.sum()
        ok0 = simulate(job, world, no_checks(), 1, rng, z=np.array([z[t]])).success[0]
        rerun_ok = simulate(job, world, no_checks(), 1, rng, z=np.array([z[t]])).success[0]
        dur = rng.lognormal(math.log(DUR["step_ms"]), 0.6, job.n)
        u = int(rng.choice(job.n, p=dur / dur.sum()))  # crash during step u (0-based), uniform in time
        effects_before = np.flatnonzero(job.irreversible[:u])
        is_effect = bool(job.irreversible[u])
        inflight = is_effect and rng.random() < DUR["inflight"]
        nondet = rng.random() < 1 - (1 - DUR["nondet"]) ** job.n
        writes_before = len(effects_before)

        # restart from scratch: every earlier side effect is repeated unless the new run notices it
        rep = rng.random(writes_before) > DUR["notice"]
        div = rng.random(writes_before) < DUR["diverge"]
        ok_scratch = rerun_ok and not rep.any() and not div.any()
        # in-memory checkpoint every 10 steps: repeat what ran since the snapshot
        s0 = (u // DUR["snapshot"]) * DUR["snapshot"]
        since = np.flatnonzero(job.irreversible[s0:u])
        rep2 = rng.random(len(since)) > DUR["notice"]
        div2 = rng.random(len(since)) < DUR["diverge"]
        seg_ok = rng.random() < math.exp(-z[t] * np.sum(np.array(world.model.hazard)[job.types[s0:u]]))
        ok_snap = seg_ok and not rep2.any() and not div2.any() and not inflight
        ok_dur = not nondet and not inflight
        ok_keys = not nondet

        ok_scratch = ok_scratch and not inflight
        for key, ok, dups in (("scratch", ok_scratch, int(rep.sum()) + int(inflight)), ("snapshot", ok_snap, int(rep2.sum()) + int(inflight)),
                              ("durable", ok_dur, int(inflight)), ("durable-keys", ok_keys, 0)):
            out[key]["dups"] += dups
            if ok0:
                out[key]["base"] += 1
                out[key]["retained"] += int(ok)
        over.append(DUR["journal_ms"] * job.n / dur.sum())
    res = {k: dict(retained=v["retained"] / v["base"], dups=100 * v["dups"] / len(jobs)) for k, v in out.items()}
    res["overhead_journal"] = float(np.mean(over))
    return seed, res


# ---------------------------------------------------------------- placement map (one back-office job)

def placement_map():
    name = "backoffice"
    jobs = build_jobs(name)
    z, scale, var = calibrate(name, jobs)
    world = make_world(name, scale, var)
    rng = np.random.default_rng([MASTER, 61])
    tr = shadow_traces(jobs[:160], world, DEV_RUNS, rng, z=z[:160])
    mdl = fit(jobs[:160], tr, template=world.model)
    lens = np.array([j.n for j in jobs])
    t = int(np.flatnonzero((lens >= 55) & (lens <= 65) & (np.arange(len(jobs)) >= 160))[0])
    job = jobs[t]
    rows = []
    allp = plans_for(job, mdl, LAMS, DMAX, Q)
    for li, lam_rel in enumerate(LAMS):
        for d in range(DMAX + 1):
            p = allp[d][li]
            rows.append(dict(lam=lam_rel, d=d, checks=p.checks, success=p.success, cost=p.cost / job.costs.sum()))
    post = [posterior_mean(mdl.frailty_var, float(np.cumsum(np.array(mdl.hazard)[job.types])[-1] / 2), d) for d in range(DMAX + 1)]
    return dict(types=job.types.tolist(), irreversible=job.irreversible.astype(int).tolist(), type_names=ENVS[name]["types"],
                check_names=[c.name for c in mdl.checks], rows=rows, posterior=post)


# ---------------------------------------------------------------- main

def mean_sd(xs):
    xs = np.asarray(xs, float)
    return [float(xs.mean()), float(xs.std(ddof=1))] if len(xs) > 1 else [float(xs.mean()), 0.0]


def frontier_at(pol, prefix, cost):
    """Success of a swept method at a given relative cost: upper hull of its sweep, linearly interpolated."""
    pts = sorted((pol[f"{prefix}-{i}"]["tokens"][0], pol[f"{prefix}-{i}"]["success"][0]) for i in range(len(LAMS)))
    front, best = [], -1.0
    for c, s in pts:  # Pareto-efficient points: better than every cheaper point
        if s > best:
            front.append((c, s))
            best = s
    xs, ys = [f[0] for f in front], [f[1] for f in front]
    if cost < xs[0] * 0.98:
        return None
    return float(np.interp(cost, xs, ys))


def half_life(ns, succ):
    """Steps at which success crosses 0.5, by log-linear interpolation (None if it never does in range)."""
    for (n0, s0), (n1, s1) in zip(zip(ns, succ), zip(ns[1:], succ[1:])):
        if s0 >= 0.5 > s1:
            t = (s0 - 0.5) / (s0 - s1)
            return float(math.exp(math.log(n0) + t * (math.log(n1) - math.log(n0))))
    return None


def aggregate_envs(env_out):
    envs = {}
    for name in ENV_ORDER:
        outs = [o for (n, s, o) in env_out if n == name]
        keys = outs[0]["policies"].keys()
        pol = {}
        for k in keys:
            pol[k] = dict(
                success=mean_sd([o["policies"][k]["success"] for o in outs]),
                tokens=mean_sd([o["policies"][k]["tokens"] for o in outs]),
                pass_k=[mean_sd([o["policies"][k]["pass_k"][i] for o in outs]) for i in range(len(outs[0]["policies"][k]["pass_k"]))],
                per_task=np.mean([o["policies"][k]["per_task"] for o in outs], axis=0).tolist(),
            )
        pred = {k: dict(success=float(np.mean([o["pred"][k]["success"] for o in outs])), pass4=float(np.mean([o["pred"][k]["pass4"] for o in outs])))
                for k in outs[0]["pred"]}
        envs[name] = dict(title=ENVS[name]["title"], policies=pol, pred=pred, fits=outs[0]["fits"], scale=outs[0]["scale"], var=outs[0]["var"],
                          tasks=ENVS[name]["tasks"], trials=ENVS[name]["trials"], retry=outs[0].get("retry"),
                          steps=[int(j.n) for j in build_jobs(name)])
    return envs


def finish(R):
    """Derived quantities, then results.json, tables, numbers and the site export."""
    for name in ENV_ORDER:
        pol = R["envs"][name]["policies"]
        R["envs"][name]["matched"] = {
            key: dict(cost=pol[key]["tokens"][0], success=pol[key]["success"][0],
                      belay=frontier_at(pol, MAIN, pol[key]["tokens"][0]), indep=frontier_at(pol, "indep", pol[key]["tokens"][0]))
            for key in pol if not key.startswith(("belay-", "static-", "indep-"))
        }
    hl = {}
    for pol in ("none", "periodic", "indep", "belay", "belay-nf", "every"):
        hl[pol] = half_life(HORIZONS, [R["horizon"][f"{n}|1.0"][pol]["success"][0] for n in HORIZONS])
    R["half_life"] = hl
    for env in R["envs"].values():  # per-task means are not reported; keep results.json small
        for p in env["policies"].values():
            p.pop("per_task", None)
    with open(os.path.join(ROOT, "results.json"), "w") as f:
        json.dump(R, f, indent=1)
    write_tables(R)
    write_fits(R)
    write_numbers(R)
    write_site(R)


def main():
    tasks = [(e, s) for e in ENV_ORDER for s in SEEDS]
    with Pool(WORKERS) as pool:
        env_out = pool.map(run_env_seed, tasks)
        hz = pool.map(horizon_point, [(n, 1.0, s) for n in HORIZONS for s in SEEDS[:3]] + [(200, v, s) for v in SYN_VARS for s in SEEDS[:3]])
        tf = pool.map(transfer_point, TRANSFER_GRID)
        du = pool.map(durability, SEEDS)
    R = {"envs": aggregate_envs(env_out), "lams": LAMS, "lam_star": LAM_STAR, "periods": PERIODS}
    H = {}
    for n, var, s, res in hz:
        H.setdefault(f"{n}|{var}", []).append(res)
    R["horizon"] = {k: {pol: dict(success=mean_sd([r[pol][0] for r in v]), tokens=mean_sd([r[pol][1] for r in v if r[pol][1] is not None])) for pol in v[0]} for k, v in H.items()}
    TF = {}
    for sh, n, s, succ, cost in tf:
        TF.setdefault(f"{sh}|{n}", []).append((succ, cost))
    R["transfer"] = {k: dict(success=mean_sd([a for a, _ in v]), tokens=mean_sd([b for _, b in v])) for k, v in TF.items()}
    D = {}
    for s, res in du:
        for k, v in res.items():
            D.setdefault(k, []).append(v)
    R["durability"] = {k: (dict(retained=mean_sd([x["retained"] for x in v]), dups=mean_sd([x["dups"] for x in v])) if isinstance(v[0], dict) else mean_sd(v)) for k, v in D.items()}
    R["placement"] = placement_map()
    finish(R)


def write_site(R):
    """The subset of results the website's companion figure draws (copied to web/src/data/papers/)."""
    star = R["lams"].index(R["lam_star"])
    envs = {}
    for e in ENV_ORDER:
        pol = R["envs"][e]["policies"]
        pt = lambda k: [round(pol[k]["tokens"][0], 3), round(100 * pol[k]["success"][0], 1)]
        envs[e] = dict(
            title=ENVS[e]["title"],
            belay=[pt(f"{MAIN}-{i}") for i in range(len(LAMS))],
            adaptive=[pt(f"belay-{i}") for i in range(len(LAMS))],
            indep=[pt(f"indep-{i}") for i in range(len(LAMS))],
            periodic=[pt(f"periodic-{T}") for T in PERIODS],
            every=pt("every"), everyFresh=pt("every-fresh"), vote=pt("vote3"), none=pt("none"),
        )
    P = R["placement"]
    plans = [[dict(checks=r["checks"], success=round(100 * r["success"], 1), cost=round(r["cost"], 2)) for r in P["rows"] if r["lam"] == lam] for lam in LAMS]
    site = dict(lams=LAMS, lamStar=R["lam_star"], star=star, periods=PERIODS, envs=envs,
                placement=dict(types=P["types"], typeNames=P["type_names"], irreversible=P["irreversible"], checkNames=P["check_names"], plans=plans),
                durability={k: dict(retained=round(100 * v["retained"][0], 1), dups=round(v["dups"][0], 1)) for k, v in R["durability"].items() if isinstance(v, dict)})
    with open(os.path.join(ROOT, "site.json"), "w") as f:
        json.dump(site, f, indent=1)


def write_numbers(R):
    """Every number the text quotes, as LaTeX macros, so the prose cannot drift from the tables."""
    star = R["lams"].index(R["lam_star"])
    out = {}
    tag = dict(retail="Ret", airline="Air", tac="Tac", backoffice="Bo")
    for e in ENV_ORDER:
        pol = R["envs"][e]["policies"]
        t = tag[e]
        for key, name in (("none", "None"), ("vote3", "Vote"), ("every", "Every"), ("every-fresh", "EveryFresh"), ("periodic-4", "Periodic"),
                          (f"indep-{star}", "Indep"), (f"static-{star}", "Belay"), (f"belay-{star}", "Adaptive")):
            out[f"{t}{name}"] = f"{100 * pol[key]['success'][0]:.1f}"
            out[f"{t}{name}Sd"] = f"{100 * pol[key]['success'][1]:.1f}"
            out[f"{t}{name}Tok"] = f"{pol[key]['tokens'][0]:.2f}"
            if len(pol[key]["pass_k"]) >= 4:
                out[f"{t}{name}PassFour"] = f"{100 * pol[key]['pass_k'][3][0]:.1f}"
        best = max(range(len(R["lams"])), key=lambda i: pol[f"{MAIN}-{i}"]["success"][0])
        out[f"{t}BelayMax"] = f"{100 * pol[f'{MAIN}-{best}']['success'][0]:.1f}"
        out[f"{t}BelayMaxTok"] = f"{pol[f'{MAIN}-{best}']['tokens'][0]:.2f}"
        m = R["envs"][e]["matched"]
        for key, name in (("every", "Every"), ("periodic-4", "Periodic"), ("vote3", "Vote"), ("every-fresh", "EveryFresh")):
            v = m[key]["belay"]
            out[f"{t}BelayAt{name}"] = "---" if v is None else f"{100 * v:.1f}"
        # cost BELAY needs to match the critic-every-step success
        xs = sorted((pol[f"{MAIN}-{i}"]["tokens"][0], pol[f"{MAIN}-{i}"]["success"][0]) for i in range(len(R["lams"])))
        target = pol["every"]["success"][0]
        hit = next((c for c, s_ in xs if s_ >= target), None)
        out[f"{t}TokToMatchEvery"] = "---" if hit is None else f"{hit:.2f}"
        out[f"{t}Saving"] = "---" if hit is None else f"{100 * (1 - (hit - 1) / (pol['every']['tokens'][0] - 1)):.0f}"
        out[f"{t}Steps"] = f"{int(np.median(R['envs'][e]['steps']))}"
        f0 = R["envs"][e]["fits"][0]
        out[f"{t}FitVar"] = f"{f0['frailty_var']:.2f}"
        out[f"{t}FitFresh"] = f"{f0['retry_fresh']:.2f}"
        out[f"{t}FitInplace"] = f"{f0['retry_inplace']:.2f}"
        out[f"{t}TrueVar"] = f"{R['envs'][e]['var']:.2f}"
    # ablations: success minus the full method's frontier at the variant's cost (points)
    for e in ("retail", "backoffice"):
        pol = R["envs"][e]["policies"]
        t = tag[e]
        for key, name in (("abl-no-frailty", "NoFrailty"), ("abl-no-latency", "NoLatency"), ("abl-inplace", "Inplace"),
                          ("abl-no-irreversible", "NoIrreversible"), ("abl-accuracy", "Accuracy"), (f"belay-{star}", "Adaptive")):
            full = frontier_at(pol, MAIN, pol[key]["tokens"][0])
            raw = 100 * (pol[key]["success"][0] - pol[f"{MAIN}-{star}"]["success"][0])
            out[f"{t}Abl{name}Raw"] = f"{raw:+.1f}"
            out[f"{t}Abl{name}Delta"] = "---" if full is None else f"{100 * (pol[key]['success'][0] - full):+.1f}"
        # BELAY against the independent-step program at matched cost, over that program's sweep
        diffs = []
        for i in range(len(R["lams"])):
            c = pol[f"indep-{i}"]["tokens"][0]
            f_ = frontier_at(pol, MAIN, c)
            if f_ is not None:
                diffs.append(100 * (f_ - pol[f"indep-{i}"]["success"][0]))
        out[f"{t}GainOverIndep"] = f"{np.mean(diffs):.1f}" if diffs else "---"
    # prediction error
    for kind, name in (("belay", "Belay"), ("indep", "Indep")):
        errs, e4 = [], []
        per_env = {}
        for e in ENV_ORDER:
            pol, pred = R["envs"][e]["policies"], R["envs"][e]["pred"]
            keys = [k.split(":", 1)[1] for k in pred if k.startswith(kind + ":")]
            ee = [pred[f"{kind}:{k}"]["success"] - pol[k]["success"][0] for k in keys]
            errs += ee
            per_env[e] = float(np.mean(np.abs(ee)))
            if e in ("retail", "airline"):
                e4 += [pred[f"{kind}:{k}"]["pass4"] - pol[k]["pass_k"][3][0] for k in keys]
        out[f"Mae{name}"] = f"{100 * np.mean(np.abs(errs)):.1f}"
        out[f"MaxErr{name}"] = f"{100 * np.max(np.abs(errs)):.1f}"
        out[f"PassFourMae{name}"] = f"{100 * np.mean(np.abs(e4)):.1f}"
        out[f"NPred"] = str(len(errs))
    # pass^k on tau-bench
    pk_rows = [("Single agent", "none"), ("Critic every step", "every"), ("Periodic, $T{=}4$", "periodic-4"), ("Indep.-step DP", f"indep-{star}"),
               ("\\textsc{Belay}", f"static-{star}"), ("\\quad+ re-planning", f"belay-{star}")]
    lines = []
    for label, key in pk_rows:
        cells = []
        for e in ("retail", "airline"):
            pk = R["envs"][e]["policies"][key]["pass_k"]
            cells += [f"{100 * pk[k - 1][0]:.1f}" for k in (1, 4, 8)]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    with open(os.path.join(ROOT, "tables", "passk.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{Retail} & \\multicolumn{3}{c}{Airline} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
                "Policy & $k{=}1$ & 4 & 8 & $k{=}1$ & 4 & 8 \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    # transfer
    T = R["transfer"]
    ns = sorted({int(k.split("|")[1]) for k in T})
    lines = []
    for sh, label in (("airline", "Retail $\\to$ airline"), ("stronger", "Stronger agent ($0.55\\times$ hazard)"), ("weaker", "Weaker agent ($1.6\\times$ hazard)")):
        cells = [f"{100 * T[f'{sh}|{m}']['success'][0]:.1f} / {T[f'{sh}|{m}']['tokens'][0]:.2f}$\\times$" for m in ns]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    with open(os.path.join(ROOT, "tables", "transfer.tex"), "w") as f:
        f.write("\\begin{center}\\small\\begin{tabular}{l" + "r" * len(ns) + "}\n\\toprule\nTarget tasks used to refit & " + " & ".join(str(m) for m in ns) + " \\\\\n\\midrule\n"
                + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\\end{center}\n"
                "Success (\\%) and tokens at $\\lambda_0$ for plans fitted on retail and refitted with $m$ target tasks pooled with 20 source tasks; 5 seeds, evaluated on the same held-out target tasks for every $m$.\n")

    # retry decomposition on back-office jobs
    for row in R["envs"]["backoffice"]["retry"]:
        nm = row["type"].capitalize()
        for key, suffix in (("first", "First"), ("fresh", "Fresh"), ("inplace", "Inplace"), ("selection_only", "Selection")):
            out[f"BoRetry{nm}{suffix}"] = f"{100 * row[key]:.1f}"
    # transfer
    T = R["transfer"]
    for sh, nm in (("airline", "Air"), ("stronger", "Strong"), ("weaker", "Weak")):
        for m, mm in ((0, "Zero"), (20, "Twenty")):
            out[f"Tr{nm}{mm}"] = f"{100 * T[f'{sh}|{m}']['success'][0]:.1f}"
            out[f"Tr{nm}{mm}Tok"] = f"{T[f'{sh}|{m}']['tokens'][0]:.2f}"
    # durability
    D = R["durability"]
    for k, name in (("scratch", "Scratch"), ("snapshot", "Snap"), ("durable", "NoKeys"), ("durable-keys", "Keys")):
        out[f"Dur{name}"] = f"{100 * D[k]['retained'][0]:.0f}"
        out[f"Dup{name}"] = f"{D[k]['dups'][0]:.1f}"
    out["DurOverhead"] = f"{100 * D['overhead_journal'][0]:.0f}"
    # horizon
    for pol, name in (("none", "None"), ("periodic", "Periodic"), ("indep", "Indep"), ("belay", "Belay"), ("belay-nf", "BelayNf"), ("every", "Every")):
        v = R["half_life"].get(pol)
        out[f"Half{name}"] = "---" if v is None else f"{v:.0f}"
        for n, word in ((100, "Hundred"), (200, "TwoHundred"), (500, "FiveHundred"), (1000, "Thousand")):
            k = f"{n}|1.0"
            if k in R["horizon"] and pol in R["horizon"][k]:
                out[f"Hz{name}{word}"] = f"{100 * R['horizon'][k][pol]['success'][0]:.1f}"
    with open(os.path.join(ROOT, "tables", "numbers.tex"), "w") as f:
        for k, v in sorted(out.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n".replace("\\\\", "\\"))
    with open(os.path.join(ROOT, "numbers.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)


# ---------------------------------------------------------------- tables

def pct(ms, bold=False, sd=True):
    m, s = ms
    body = f"{100 * m:.1f}" + (f"\\,$\\pm$\\,{100 * s:.1f}" if sd else "")
    return f"\\textbf{{{body}}}" if bold else body


def tok(ms, bold=False):
    body = f"{ms[0]:.2f}$\\times$"
    return f"\\textbf{{{body}}}" if bold else body


def write_fits(R):
    """Appendix: fitted hazards and check parameters (first fold, seed 0), one block per environment."""
    blocks = []
    tex_title = dict(retail="$\\tau$-bench retail", airline="$\\tau$-bench airline", tac="TheAgentCompany", backoffice="Back-office jobs")
    for e in ENV_ORDER:
        env, f = ENVS[e], R["envs"][e]["fits"][0]
        types = env["types"]
        k = len(types)
        rows = [f"\\multicolumn{{{k + 3}}}{{l}}{{\\textbf{{{tex_title[e]}}}}} \\\\",
                " & " + " & ".join(types) + " & $r^\\infty/r^0$ & $f$ \\\\",
                "\\cmidrule(lr){2-" + str(k + 3) + "}",
                "Hazard $\\hat\\eta$ & " + " & ".join(f"{h:.3f}" for h in f["hazard"]) + " & & \\\\",
                "Independent fit & " + " & ".join(f"{h:.3f}" for h in f["indep_hazard"]) + " & & \\\\"]
        for c, spec in zip(f["checks"], env["checks"]):
            cells = []
            for t in range(k):
                visible = spec.after is None or t in spec.after or c["recall_inf"] > 0.05
                cells.append(f"{c['recall0'] * c['coverage'][t]:.2f}" if visible else "--")
            ratio = c["recall_inf"] / max(c["recall0"], 1e-9)
            rows.append(f"Check: {c['name']} & " + " & ".join(cells) + f" & {ratio:.2f} & {c['false_flag']:.3f} \\\\")
        blocks.append((k, "\n".join(rows)))
    width = max(k for k, _ in blocks) + 3
    body = "\n\\midrule\n".join(b for _, b in blocks)
    with open(os.path.join(ROOT, "tables", "fits.tex"), "w") as fh:
        fh.write("\\begin{tabular}{l" + "r" * (width - 1) + "}\n\\toprule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")


def write_tables(R):
    star = R["lams"].index(R["lam_star"])
    rows = [
        ("Single agent (no checks)", "none"),
        ("Majority vote of 3 per step", "vote3"),
        ("Critic after every step, retry in place", "every"),
        ("Periodic critic, $T{=}4$, rollback", "periodic-4"),
        ("Independent-step DP", f"indep-{star}"),
        ("\\textsc{Belay}", f"static-{star}"),
        ("\\textsc{Belay} + re-planning after flags", f"belay-{star}"),
    ]
    lines = []
    for label, key in rows:
        cells = []
        for e in ENV_ORDER:
            p = R["envs"][e]["policies"][key]
            ours = key.startswith("static")
            cells += [pct(p["success"], bold=ours), tok(p["tokens"])]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    tex_title = dict(retail="$\\tau$-bench retail", airline="$\\tau$-bench airline", tac="TheAgentCompany", backoffice="Back-office jobs")
    hdr = "\\begin{tabular}{l" + "rr" * 4 + "}\n\\toprule\n & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{tex_title[e]}}}" for e in ENV_ORDER) + " \\\\\n"
    hdr += " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(4)) + "\n"
    hdr += "Policy & " + " & ".join(["Succ.\\ (\\%)", "Tok."] * 4) + " \\\\\n\\midrule\n"
    with open(os.path.join(ROOT, "tables", "main.tex"), "w") as f:
        f.write(hdr + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    # ablations
    abl = [("\\textsc{Belay}", f"static-{star}"), ("$-$ frailty", "abl-no-frailty"), ("$+$ re-planning after flags", f"belay-{star}"),
           ("$-$ recall by latency", "abl-no-latency"), ("Retry in place", "abl-inplace"),
           ("$-$ irreversibility", "abl-no-irreversible"), ("Checks ranked by accuracy", "abl-accuracy")]
    lines = []
    for label, key in abl:
        cells = []
        for e in ("retail", "backoffice"):
            p = R["envs"][e]["policies"][key]
            cells += [pct(p["success"], sd=False), tok(p["tokens"])]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    with open(os.path.join(ROOT, "tables", "ablation.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n & \\multicolumn{2}{c}{$\\tau$-bench retail} & \\multicolumn{2}{c}{Back-office} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nVariant & Succ. & Tok. & Succ. & Tok. \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    # durability
    D = R["durability"]
    names = [("Restart from scratch", "scratch", "---"), ("Snapshot every 10 steps", "snapshot", "1\\%"),
             ("Journal, no keys", "durable", f"{100 * D['overhead_journal'][0]:.0f}\\%"), ("Journal + idempotency keys", "durable-keys", f"{100 * D['overhead_journal'][0]:.0f}\\%")]
    lines = [f"{lab} & {pct(D[k]['retained'], bold=k == 'durable-keys', sd=False)} & {D[k]['dups'][0]:.1f} & {ov} \\\\" for lab, k, ov in names]
    with open(os.path.join(ROOT, "tables", "durability.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrr}\n\\toprule\nRecovery & Kept (\\%) & Dup. & Time \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    if "--transfer-only" in sys.argv:  # recompute the transfer study into an existing results.json
        R = json.load(open(os.path.join(ROOT, "results.json")))
        with Pool(WORKERS) as pool:
            tf = pool.map(transfer_point, TRANSFER_GRID)
        TF = {}
        for sh, n, s_, succ, cost in tf:
            TF.setdefault(f"{sh}|{n}", []).append((succ, cost))
        R["transfer"] = {k: dict(success=mean_sd([a for a, _ in v]), tokens=mean_sd([b for _, b in v])) for k, v in TF.items()}
        with open(os.path.join(ROOT, "results.json"), "w") as f:
            json.dump(R, f, indent=1)
        write_tables(R)
        write_numbers(R)
        write_site(R)
    elif "--envs-only" in sys.argv:  # recompute the four benchmarks and the placement map, keep the rest
        R = json.load(open(os.path.join(ROOT, "results.json")))
        with Pool(WORKERS) as pool:
            env_out = pool.map(run_env_seed, [(e, s_) for e in ENV_ORDER for s_ in SEEDS])
        R["envs"] = aggregate_envs(env_out)
        R["placement"] = placement_map()
        finish(R)
    elif "--tables-only" in sys.argv:  # rewrite tables, numbers and the site export from an existing results.json
        finish(json.load(open(os.path.join(ROOT, "results.json"))))
    else:
        main()
