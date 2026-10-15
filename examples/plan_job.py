"""Plan where to check a long job of your own with BELAY.

    python examples/plan_job.py examples/data/ap_job.yaml

The job file names the step types, the step sequence (a step ending in ``!`` is irreversible), the per-step
derailment hazard of each type, the task frailty, the retry multipliers and the checks available, with their cost,
recall and false-flag rate. ``belay.fit`` estimates these from shadow-mode traces of your agent; here they are
written down. For each price (success given up per unchecked run of tokens) the script prints the plan the
Lagrangian program finds, with the model's success and token predictions, and simulates two simple policies under
the same model for comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from belay.model import Check, Job, RunModel
from belay.placement import solve_linear
from belay.simulate import World, every_step, no_checks, simulate


def load(path: str) -> tuple[Job, RunModel, list[str]]:
    spec = yaml.safe_load(Path(path).read_text())
    kinds = spec["step_types"]
    steps = spec["steps"].split()
    types = np.array([kinds.index(s.rstrip("!")) for s in steps])
    job = Job(types=types, costs=np.ones(len(steps)), irreversible=np.array([s.endswith("!") for s in steps]))
    checks = []
    for c in spec["checks"]:
        c = dict(c)
        if "coverage" in c:
            c["coverage"] = tuple(c["coverage"][k] for k in kinds)
        if "after" in c:
            c["after"] = tuple(kinds.index(k) for k in c["after"])
        checks.append(Check(**c))
    model = RunModel(hazard=tuple(spec["hazard"][k] for k in kinds), checks=tuple(checks), **spec.get("run", {}))
    return job, model, kinds


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("job", help="YAML file describing the job and its checks")
    ap.add_argument("--prices", type=float, nargs="*", default=[0.01, 0.05, 0.2])
    ap.add_argument("--runs", type=int, default=4000, help="simulated runs for the comparison policies")
    args = ap.parse_args(argv)
    job, model, kinds = load(args.job)
    unchecked = float(job.costs.sum())
    counts = Counter(kinds[t] for t in job.types)
    print(f"job: {job.n} steps ({', '.join(f'{n} {k}' for k, n in counts.items())}; {int(job.irreversible.sum())} irreversible), "
          f"checks: {', '.join(c.name for c in model.checks)}")
    print(f"{'price':>6}{'success':>9}{'tokens':>8}  checks placed")
    plans = {}
    for lam in args.prices:
        plan = solve_linear(job, model, lam0=lam)
        placed = Counter(model.checks[v].name for _, v in plan.checks)
        print(f"{lam:>6.2f}{plan.success:>9.3f}{plan.cost / unchecked:>7.2f}x  "
              + (", ".join(f"{name} x{n}" for name, n in placed.items()) or "none"))
        plans[lam] = plan
    world = World(model, frailty="gamma")
    rng = np.random.default_rng(0)
    critic = next(i for i, c in enumerate(model.checks) if c.name == "critic")
    for name, policy in (("no checks", no_checks()), ("critic after every step", every_step(critic, mode="inplace"))):
        res = simulate(job, world, policy, args.runs, rng)
        print(f"{name} (simulated under the same model): success {res.success.mean():.3f} "
              f"at {res.tokens.mean() / unchecked:.2f}x tokens")
    last = plans[args.prices[-1]]
    where = ", ".join(f"{model.checks[v].name} after step {b}" for b, v in last.checks[:6])
    print(f"first checks at price {args.prices[-1]}: {where}{', ...' if len(last.checks) > 6 else ''}")
    return {"plans": plans}


if __name__ == "__main__":
    main()
