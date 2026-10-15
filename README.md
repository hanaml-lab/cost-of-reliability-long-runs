# BELAY: budget-optimal placement of verifiers and rollbacks

[![ci](https://github.com/hanaml-lab/cost-of-reliability-long-runs/actions/workflows/ci.yml/badge.svg)](https://github.com/hanaml-lab/cost-of-reliability-long-runs/actions/workflows/ci.yml)

Reference implementation of *Where to Check: Budget-Optimal Placement of Verifiers and Rollbacks for Long-Horizon Agents* (Agarwal, hanaML technical report, 2025).

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
pip install -e ".[dev]"      # numpy, scipy, pyyaml; pytest and ruff
pip install -e ".[paper]"    # + matplotlib, to regenerate the report's tables and figures
```

## Quickstart (CPU, under 20 seconds)

```bash
python -m belay.demo         # or belay-demo; settings in configs/toy.yaml
pytest -q                    # includes the program against exhaustive enumeration
```

## Toy run (real output)

The toy has 100 jobs of 20-40 steps with three step types (a reliable read, a risky decision and an irreversible write) and two checks (a near-free rule check and a same-model critic). Its ground truth has task frailty, an upward hazard drift and a critic blind on hard tasks, effects the planner's model partly omits. The planner fits its model on shadow-mode traces of half the jobs and every policy is simulated on the other half. Output of `python -m belay.demo` on a CPU (Python 3.12):

```
50 held-out toy jobs, 24 simulated runs each; price 0.02 per run of tokens
policy                           success   tokens
no checks                          0.470    1.00x
critic after every step            0.612    2.06x
critic every 4 steps               0.532    1.34x
independent-step program           0.580    1.47x
BELAY                              0.639    1.39x
BELAY, re-planning after flags     0.644    1.39x
done in 7.7s
```

Of the static policies, BELAY's plan succeeds most often, at two-thirds of the tokens of a critic after every step. The independent-step program, the same program on a fit without frailty, latency-dependent recall or retry modes, spends more for less. `paper/toy_run.md` records the environment and what each line shows.

## Planning your own job

`examples/plan_job.py` reads a job described in YAML (its step types and sequence, irreversible steps, per-type hazards, retry multipliers, and the checks available with their cost, recall and false-flag rate; `belay.fit` estimates these from shadow-mode traces) and prints the plan at several prices.

```bash
python examples/plan_job.py examples/data/ap_job.yaml
```

```
job: 30 steps (12 read, 6 match, 6 decide, 6 pay; 6 irreversible), checks: totals, critic
 price  success  tokens  checks placed
  0.01    0.846   1.61x  critic x18, totals x12
  0.05    0.843   1.43x  critic x12, totals x12
  0.20    0.807   1.04x  totals x12
no checks (simulated under the same model): success 0.639 at 1.00x tokens
critic after every step (simulated under the same model): success 0.781 at 2.02x tokens
first checks at price 0.2: totals after step 3, totals after step 5, totals after step 8, totals after step 10, totals after step 13, totals after step 15, ...
```

The same from Python:

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

These numbers come from the report's seeded simulation and are illustrative. Qwen3-32B agent; success (%) and tokens relative to a single run at the default price λ₀ = 0.2, mean over five seeds.

| Policy | τ-bench retail | Tokens | Back-office | Tokens |
|---|---|---|---|---|
| Single agent | 52.2 | 1.00× | 48.4 | 1.00× |
| Critic after every step | 64.9 | 2.08× | 61.8 | 2.06× |
| Periodic critic, T = 4 | 58.3 | 1.36× | 54.0 | 1.33× |
| Independent-step program | 60.5 | 1.13× | 64.5 | 1.16× |
| **BELAY** | **63.6** | **1.22×** | **67.9** | **1.21×** |

Model prediction on held-out policies: mean absolute error 0.8 points (independent-step fit: 4.5); pass^4 1.0 (independent-step fit: 16.4). Durable journal with idempotency keys: 97% of successful jobs survive an injected crash, against 38% for restarting an in-memory loop.

`bash scripts/reproduce_paper.sh` regenerates every table and figure (about ten minutes on 16 cores; `WORKERS` sets the number of processes). The report's tables were generated with SciPy linked against Intel MKL. This simulation fits its models with SciPy's optimisers, so it is sensitive to the BLAS build: with the PyPI wheels (OpenBLAS) the regenerated tables differ from the report's by at most 1.2 points, and by 0.1 on the headline row (for example BELAY on τ-bench retail, 63.5 instead of 63.6). A figure derived from them moves further: at the success of a critic after every step, BELAY saves 55% of the extra tokens rather than the report's 65%. CI regenerates the tables in a pinned environment (numpy 2.3.5, SciPy 1.16.3, one OpenBLAS kernel and thread) and checks them against `paper/results.sha256`, which records that environment's output; `python paper/check_results.py` runs the same check.

## Layout

```
src/belay/      model.py  placement.py  adaptive.py  fit.py  simulate.py  theory.py  journal.py  toy.py  demo.py
configs/        toy.yaml  tau_bench.yaml  theagentcompany.yaml  backoffice.yaml
examples/       plan_job.py, data/ap_job.yaml
scripts/        run_toy.py  reproduce_paper.sh
paper/          make_data.py  figures.py  check_results.py  results.sha256  toy_run.md
tests/          pytest suite
```

## Full-scale runs

The configs record the report's settings: a Qwen3-32B ReAct agent on τ-bench (retail and airline), TheAgentCompany and the back-office jobs, with each environment's deterministic checks and a same-model critic. This repository holds the model, the placement program, fitting and simulation; it does not include agent scaffolds, benchmark harnesses or model weights.

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

## License

Apache-2.0. See `LICENSE`.
