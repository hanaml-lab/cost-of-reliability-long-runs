"""Algorithm 1 of the report: budget-optimal placement of checks and rollback points.

A plan is a set of check positions b_1 < ... < b_J with a check type at each; a passed check is the
rollback point for the next segment. For a token price lam the program maximises

    sum_j [ log E[S_j | on track at b_{j-1}] - lam * E[C_j | on track at b_{j-1}] ]  +  tail term,

where S_j and C_j are the success and cost of segment j with its retries (``belay.model``) and the
expectation is over the task's frailty Z given that the run reached b_{j-1} on track: a gamma prior
tilted by exp(-Z H(b_{j-1})) (and by Z^d after d detections, the conjugate update for the adaptive
policy). The sum of log conditional successes is the log of the run's success, so the program is a
longest-path problem over rollback points, solved backwards in O(n * W * |checks| * Q) for a window of
W steps and Q frailty nodes. Sweeping lam traces the cost-reliability frontier.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Job, RunModel, frailty_nodes


@dataclass
class Plan:
    """A placement: checks as (position, check index), and the next check from every rollback point."""

    checks: list[tuple[int, int]]
    objective: float
    success: float
    cost: float
    lam: float
    next_b: np.ndarray  # next_b[a]: the next check position after a passed check at a (n: none)
    next_v: np.ndarray  # next_v[a]: its check index, -1 for "run to the end unchecked"


def _survivor_weights(z: np.ndarray, w: np.ndarray, H: np.ndarray, detections: int) -> np.ndarray:
    """Posterior weights over frailty nodes at every position a, given on track at a: shape (n+1, Q)."""
    logw = np.log(w)[None, :] + detections * np.log(z)[None, :] - np.outer(H, z)
    logw -= logw.max(axis=1, keepdims=True)
    out = np.exp(logw)
    return out / out.sum(axis=1, keepdims=True)


def _detect_tables(job: Job, model: RunModel, mu: float, H: np.ndarray, eta: np.ndarray, window: int) -> np.ndarray:
    """D[v, a, j]: probability that steps a+1..a+j+1 derail and check v at a+j+1 catches it."""
    n = job.n
    V = len(model.checks)
    out = np.zeros((V, n, window))
    A = np.arange(n)
    for v, chk in enumerate(model.checks):
        cov = np.array([chk.covers(t) for t in range(len(model.hazard))])[job.types]
        r0, rinf, phi = chk.recall0, chk.r_inf, chk.decay
        G = np.zeros(n)
        acc = np.zeros(n)
        for j in range(window):
            i = A + j + 1
            ok = i <= n
            ic = np.minimum(i, n)
            surv_prev = np.exp(-mu * (H[ic - 1] - H[A]))
            g = np.where(ok, surv_prev * (1 - np.exp(-mu * eta[ic - 1])) * cov[ic - 1], 0.0)
            G = G + g
            acc = phi * acc + g
            out[v, :, j] = rinf * G - (rinf - r0) * acc
    return out


def segment_tables(job: Job, model: RunModel, *, q: int = 16, detections: int = 0, mode: str = "fresh", window: int | None = None):
    """Frailty-averaged segment success and cost for every (a, b = a+j+1, check v).

    Returns (S, C, tail_S, tail_C, valid) with S, C, valid of shape (n, W, V) and the tails of shape (n,).
    """
    n, V = job.n, len(model.checks)
    W = min(n, window or 64)
    eta = model.eta(job)
    H = np.concatenate([[0.0], np.cumsum(eta)])
    Cc = np.concatenate([[0.0], np.cumsum(job.costs)])
    L = job.last_commit_before()
    z, w = frailty_nodes(model.frailty_var, q)
    post = _survivor_weights(z, w, H, detections)  # (n+1, Q)

    A = np.arange(n)[:, None]
    B = A + 1 + np.arange(W)[None, :]
    valid2 = B <= n
    Bc = np.minimum(B, n)
    Lp = np.maximum(A, L[Bc])
    jr = np.clip(Bc - Lp - 1, 0, W - 1)
    cost_b = job.costs[Bc - 1]
    rho = model.retry_multiplier(mode)
    R = model.max_retries

    applies = np.array([[chk.applies(t) for t in job.types] for chk in model.checks])  # (V, n)
    valid = valid2[:, :, None] & applies[:, Bc - 1].transpose(1, 2, 0)
    S = np.zeros((n, W, V))
    C = np.zeros((n, W, V))
    tail_S = np.zeros(n)
    for qi, zq in enumerate(z):
        wq = post[:n, qi][:, None]
        D1 = _detect_tables(job, model, zq, H, eta, W)
        Dr = D1 if rho == 1.0 else _detect_tables(job, model, zq * rho, H, eta, W)
        p1 = np.exp(-zq * (H[Bc] - H[A]))
        ppre = np.exp(-zq * (H[Lp] - H[A]))
        pr = np.exp(-zq * rho * (H[Bc] - H[Lp]))
        for v, chk in enumerate(model.checks):
            f = chk.false_flag
            k = chk.cost * cost_b + chk.fixed_cost
            d1 = ppre * D1[v][Lp, jr]
            dr = Dr[v][Lp, jr]
            ok1, flag1 = p1 * (1 - f), p1 * f + d1
            okr, flagr = pr * (1 - f), pr * f + dr
            c1 = Cc[Bc] - Cc[A] + k
            cr = Cc[Bc] - Cc[Lp] + k + (model.rollback_cost * cost_b if mode == "fresh" else 0.0)
            if R == 0:
                s, c = ok1 + p1 * f, c1
            else:
                geo = np.where(flagr < 1, (1 - flagr**R) / np.maximum(1 - flagr, 1e-300), R)
                s = ok1 + flag1 * okr * geo + flag1 * flagr ** (R - 1) * pr * f
                c = c1 + flag1 * cr * geo
            S[:, :, v] += wq * s
            C[:, :, v] += wq * c
        tail_S += post[:n, qi] * np.exp(-zq * (H[n] - H[:n]))
    tail_C = Cc[n] - Cc[:n]
    return S, C, tail_S, tail_C, valid


def solve(job: Job, model: RunModel, lam: float, *, q: int = 16, detections: int = 0, mode: str = "fresh", window: int | None = None) -> Plan:
    """The placement maximising log-success minus lam times expected tokens (Algorithm 1)."""
    n = job.n
    S, C, tail_S, tail_C, valid = segment_tables(job, model, q=q, detections=detections, mode=mode, window=window)
    W, V = S.shape[1], S.shape[2]
    with np.errstate(divide="ignore"):
        logS = np.where(valid, np.log(np.maximum(S, 1e-300)), -np.inf)
        tail = np.log(np.maximum(tail_S, 1e-300)) - lam * tail_C
    score = logS - lam * C
    value = np.zeros(n + 1)
    next_b = np.full(n + 1, n, dtype=int)
    next_v = np.full(n + 1, -1, dtype=int)
    succ = np.ones(n + 1)
    cost = np.zeros(n + 1)
    for a in range(n - 1, -1, -1):
        jmax = min(W, n - a)
        cand = score[a, :jmax, :] + value[a + 1 : a + 1 + jmax, None]
        flat = int(np.argmax(cand))
        j, v = divmod(flat, V)
        if cand[j, v] > tail[a]:
            value[a] = cand[j, v]
            next_b[a], next_v[a] = a + j + 1, v
            succ[a] = S[a, j, v] * succ[a + j + 1]
            cost[a] = C[a, j, v] + cost[a + j + 1]
        else:
            value[a] = tail[a]
            succ[a], cost[a] = tail_S[a], tail_C[a]
    checks, a = [], 0
    while a < n and next_v[a] >= 0:
        checks.append((int(next_b[a]), int(next_v[a])))
        a = next_b[a]
    return Plan(checks=checks, objective=float(value[0]), success=float(succ[0]), cost=float(cost[0]), lam=lam, next_b=next_b, next_v=next_v)


def frontier(job: Job, model: RunModel, lams, **kwargs) -> list[Plan]:
    """Plans for a sweep of token prices: the supported points of the cost-reliability frontier."""
    return [solve(job, model, lam, **kwargs) for lam in lams]
