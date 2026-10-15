# Toy run (real output)

A real run of the reference implementation on its CPU toy setting. Unlike the report's tables, which come from a
seeded simulation and are illustrative, every number below was printed by the code. The toy is small by design: it
checks that the method works end to end on a problem with a known answer; it does not reproduce the report's
experiments.

- Command: `python -m belay.demo` (settings in `configs/toy.yaml`)
- Code: commit `d7a7f3c`
- Environment: Python 3.12.13, numpy 2.5.4, scipy 1.18.1, PyYAML 6.0.3; Linux x86_64, Intel(R) Core(TM) i9-14900KF
- Wall time: 14 s

## The toy setting

`belay.toy` makes 100 jobs of 20 to 40 steps with three step types: a reliable read, a risky decision and an
irreversible write. Two checks are available: a near-free rule check that sees some errors at the step it follows,
and a same-model critic. The ground truth has gamma task frailty, an upward hazard drift and a critic that is blind
on hard tasks; the planner's model has no drift and no blind spot, so it is misspecified, as in practice. The
planner fits its model on shadow-mode traces of half the jobs (8 runs each) and is evaluated on the other 50 jobs,
24 simulated runs each, at a price of 0.02 success per unchecked run of tokens. The policies are no checks, a critic
after every step (retry in place), a critic every 4 steps, the independent-step program (the same program on a fit
without frailty, latency-dependent recall or retry modes), BELAY's static plan, and BELAY re-planning after flags.

## Output

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

## What it shows

- Of the static policies, BELAY's plan succeeds most often (0.639), at 1.39 times the tokens of a single run. A
  critic after every step succeeds less often (0.612) at 2.06 times.
- The independent-step program spends more (1.47 times) for less (0.580). It is the same program on a simpler
  fit, so the gap is what fitting frailty, latency-dependent recall and retry modes buys.
- Re-planning after flags adds half a point (0.644 at the same cost); the report finds about a point, and keeps the
  static plan as the default.
- A critic every 4 steps is cheap (1.34 times) but adds only 6 points over no checks (0.532 against 0.470).
