# belay: budget-optimal placement of verifiers and rollbacks

Reference implementation for **"Where to Check: Budget-Optimal Placement of Verifiers and Rollbacks for Long-Horizon Agents"** (hanaML technical report, December 2025).

A long agent run fails by compounding, and the usual remedies (critics, retries, voting, re-planning) cost tokens wherever they run. BELAY decides where to spend them. It models a run as a hidden derailment process and solves for the plan that maximises expected success minus a price on tokens:

- **The derailment model** (`belay.model`): step hazards by type, a task-level gamma frailty shared by every run of a task, checks with a cost, a false-flag rate, per-type coverage and a recall that depends on how many steps ago the error happened, fresh-rollback and in-place retries with their own hazard multipliers, and irreversible steps that no rollback can cross.
- **The placement program** (`belay.placement`, Algorithm 1): a Lagrangian dynamic program over rollback points, exact without frailty and O(n·W·V·Q) for n steps, a window of W, V check types and Q frailty nodes. `solve_sweep` traces the cost–reliability frontier from one table build; `solve_linear` solves the linear objective E[success] − λ₀·E[tokens]/C₀.
- **Re-planning after flags** (`belay.adaptive`): one plan per number of flags, from the Poisson–gamma posterior on the task's frailty. In the report it changes success by about a point; the static plan is the default.
- **Fitting** (`belay.fit`): maximum likelihood from shadow-mode traces, with one frailty draw per task, recall by error type and latency, and retry multipliers that separate contamination from selection. `frailty=False, latency=False, retry_modes=False` gives the independent-step fit used as the baseline model.
- **Simulation** (`belay.simulate`): a Monte Carlo simulator of runs under any plan, with ground-truth effects the model omits (hazard drift, a critic blind on hard tasks, correlated votes).
- **Closed forms** (`belay.theory`): break-even recall for a check, retry success under frailty, Young's check interval, and the half-life multiplier of checking.
- **Durable execution** (`belay.journal`): a minimal journalled executor with deterministic replay and idempotency keys, to test the guarantees directly.

## Install

```bash
pip install -e ".[dev]"      # numpy, scipy, pytest
pip install -e ".[paper]"    # + matplotlib, to regenerate the report's tables and figures
```

## Quickstart (CPU, under ten seconds)

```bash
python scripts/run_toy.py    # fit on shadow traces, plan, and compare policies on 50 held-out toy jobs
pytest -q                    # 36 tests, including the program against exhaustive enumeration
```

```
policy         success   tokens
none             0.470    1.00x
every            0.612    2.06x
periodic         0.532    1.34x
indep            0.580    1.47x
belay            0.639    1.39x
belay+replan     0.644    1.39x
```

### Planning your own job

```python
import numpy as np
from belay.model import Check, Job, RunModel
from belay.placement import solve_linear

critic = Check("critic", cost=0.8, recall0=0.62, recall_inf=0.27, decay=0.55, false_flag=0.07)
rule = Check("totals", cost=0.0, fixed_cost=0.03, recall0=0.97, false_flag=0.005, coverage=(0.0, 0.7, 0.6), after=(1, 2))
model = RunModel(hazard=(0.004, 0.05, 0.02), checks=(rule, critic), frailty_var=1.5, retry_fresh=0.9, retry_inplace=1.6)
job = Job(types=np.array([0, 1, 2] * 10), costs=np.ones(30), irreversible=np.array([False, False, True] * 10))
plan = solve_linear(job, model, lam0=0.2)       # λ₀: success given up per unchecked run of tokens
print(plan.checks, plan.success, plan.cost)     # [(position, check index), ...]
```

## Results (from the report)

Qwen3-32B agent; success (%) and tokens relative to a single run at the default price λ₀ = 0.2, mean over five seeds.

| Policy | τ-bench retail | Tokens | Back-office | Tokens |
|---|---|---|---|---|
| Single agent | 52.2 | 1.00× | 48.4 | 1.00× |
| Critic after every step | 64.9 | 2.08× | 61.8 | 2.06× |
| Periodic critic, T = 4 | 58.3 | 1.36× | 54.0 | 1.33× |
| Independent-step program | 60.5 | 1.13× | 64.5 | 1.16× |
| **BELAY** | **63.6** | **1.22×** | **67.9** | **1.21×** |

Model prediction on held-out policies: mean absolute error 0.8 points (independent-step fit: 4.5); pass^4 1.0 (independent-step fit: 16.4). Durable journal with idempotency keys: 97% of successful jobs survive an injected crash, against 38% for restarting an in-memory loop.

The report's numbers come from a seeded simulation study calibrated to public baselines (see its Limitations section). `scripts/reproduce_paper.sh` regenerates every table and figure from `paper/make_data.py` and `paper/figures.py`.

## Layout

```
src/belay/      model.py  placement.py  adaptive.py  fit.py  simulate.py  theory.py  journal.py  toy.py
configs/        toy.toml  tau_bench.toml  theagentcompany.toml  backoffice.toml
scripts/        run_toy.py  reproduce_paper.sh
paper/          make_data.py  figures.py
tests/          pytest suite
```

## Full-scale runs

The configs record the paper's settings: a Qwen3-32B ReAct agent on τ-bench (retail and airline), TheAgentCompany and the back-office jobs, with each environment's deterministic checks and a same-model critic. This repository holds the model, the placement program, fitting and simulation; it does not include agent scaffolds, benchmark harnesses or model weights.

## Citation

```bibtex
@techreport{agarwal2025belay,
  title       = {Where to Check: Budget-Optimal Placement of Verifiers and Rollbacks for Long-Horizon Agents},
  author      = {Agarwal, Manav},
  institution = {hanaML},
  type        = {Technical report},
  year        = {2025},
  url         = {https://hanaml.com/research/papers/cost-of-reliability-long-runs}
}
```

## Licence

Apache-2.0.
