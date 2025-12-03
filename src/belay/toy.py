"""A CPU toy setting that runs the whole pipeline in a few seconds.

A hundred jobs of 20-40 steps with three step types (a reliable read, a risky decision and an irreversible
write) and two checks (a near-free rule check that sees some errors at the step it follows, and a
same-model critic). The ground truth has gamma task frailty, an upward hazard drift and a critic that is
blind on hard tasks; the planner fits its model on shadow-mode traces of half the jobs and is evaluated
on the other half against no checks, a critic after every step, a periodic critic and the
independent-step program; "belay+replan" is the variant that re-plans after flags.
"""
from __future__ import annotations

import numpy as np

from .adaptive import adaptive_plans
from .fit import fit, shadow_traces
from .model import Check, Job, RunModel
from .placement import solve_linear
from .simulate import World, adaptive, every_step, no_checks, periodic, simulate, static

CHECKS = (
    Check("rule", cost=0.0, fixed_cost=0.03, recall0=0.95, recall_inf=0.0, decay=0.0, false_flag=0.005, coverage=(0.6, 0.3, 0.5), after=(0, 2)),
    Check("critic", cost=0.8, recall0=0.62, recall_inf=0.27, decay=0.55, false_flag=0.07),
)
TRUTH = RunModel(hazard=(0.01, 0.09, 0.04), checks=CHECKS, frailty_var=1.2, retry_fresh=0.85, retry_inplace=1.9, max_retries=2, rollback_cost=0.35)


def toy_jobs(count: int = 100, seed: int = 0) -> list[Job]:
    rng = np.random.default_rng(seed)
    jobs = []
    for _ in range(count):
        seq = []
        while len(seq) < int(rng.integers(20, 41)):
            seq += [0] * int(rng.integers(1, 3)) + [1] + ([2] if rng.random() < 0.6 else [])
        types = np.array(seq, dtype=int)
        jobs.append(Job(types=types, costs=1 + 0.6 * np.arange(len(types)) / max(len(types) - 1, 1), irreversible=types == 2))
    return jobs


def run_toy(seed: int = 0, lam0: float = 0.02, trials: int = 24) -> dict:
    rng = np.random.default_rng(seed)
    world = World(TRUTH, frailty="gamma", drift=0.25, blind_spot=(0.0, 0.35))
    jobs = toy_jobs(seed=seed)
    z = world.sample_z(rng, len(jobs))
    dev, test = np.arange(0, len(jobs), 2), np.arange(1, len(jobs), 2)
    traces = shadow_traces([jobs[i] for i in dev], world, 8, rng, z=z[dev])
    model = fit([jobs[i] for i in dev], traces, template=TRUTH)
    indep = fit([jobs[i] for i in dev], traces, template=TRUTH, frailty=False, latency=False, retry_modes=False)
    out = {k: dict(success=[], tokens=[]) for k in ("none", "every", "periodic", "indep", "belay", "belay+replan")}
    for t in test:
        job, zz = jobs[t], np.full(trials, z[t])
        policies = {
            "none": no_checks(),
            "every": every_step(1, mode="inplace"),
            "periodic": periodic(1, every=4),
            "indep": static(solve_linear(job, indep, lam0, q=1)),
            "belay": static(solve_linear(job, model, lam0, q=12)),
            "belay+replan": adaptive(adaptive_plans(job, model, lam0, max_detections=3, linear=True, q=12)),
        }
        for name, policy in policies.items():
            res = simulate(job, world, policy, trials, rng, z=zz)
            out[name]["success"].append(res.success.mean())
            out[name]["tokens"].append(res.tokens.mean() / job.costs.sum())
    return {k: dict(success=float(np.mean(v["success"])), tokens=float(np.mean(v["tokens"]))) for k, v in out.items()}
