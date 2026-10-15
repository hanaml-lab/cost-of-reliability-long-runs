"""All figures of the report, drawn from results.json (written by make_data.py) at their printed size.

ICML two-column layout: a column is 3.25 in wide and a full-width figure 6.75 in; fonts are 7-8 pt.
"""
import json
import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ROOT, "results.json")))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "grid.linewidth": 0.4, "grid.alpha": 0.35,
    "axes.grid": True, "lines.linewidth": 1.1, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})
# Okabe-Ito
C = dict(belay="#0072B2", static="#56B4E9", indep="#E69F00", periodic="#009E73", every="#D55E00", fresh="#CC79A7", vote="#999999", none="#000000")
ENVS = ["retail", "airline", "tac", "backoffice"]
TITLE = {"retail": r"$\tau$-bench retail", "airline": r"$\tau$-bench airline", "tac": "TheAgentCompany", "backoffice": "Back-office jobs"}
LAMS = R["lams"]
STAR = LAMS.index(R["lam_star"])
COL, FULL = 3.25, 6.75


def sweep(pol, prefix, n):
    x = np.array([pol[f"{prefix}-{i}"]["tokens"][0] for i in range(n)])
    y = np.array([pol[f"{prefix}-{i}"]["success"][0] for i in range(n)]) * 100
    e = np.array([pol[f"{prefix}-{i}"]["success"][1] for i in range(n)]) * 100
    o = np.argsort(x)
    return x[o], y[o], e[o]


def front(x, y):
    """Indices of the efficient points of a sweep sorted by cost: each better than every cheaper point."""
    keep, best = [], -1.0
    for i in range(len(x)):
        if y[i] > best:
            keep.append(i)
            best = y[i]
    return np.array(keep)


# 1. cost-reliability frontiers -------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(FULL, 1.85))
for ax, env in zip(axs, ENVS):
    pol = R["envs"][env]["policies"]
    x, y, e = sweep(pol, "static", len(LAMS))
    f = front(x, y)
    ax.fill_between(x[f], (y - 1.96 * e / math.sqrt(5))[f], (y + 1.96 * e / math.sqrt(5))[f], color=C["belay"], alpha=0.18, lw=0)
    ax.plot(x[f], y[f], "-o", ms=2.2, color=C["belay"], lw=1.4, zorder=5, label="BELAY")
    ax.plot(x, y, "o", ms=2.2, color=C["belay"], zorder=5)
    x, y, _ = sweep(pol, "belay", len(LAMS))
    f = front(x, y)
    ax.plot(x[f], y[f], "-", color=C["static"], lw=0.9, label="BELAY + re-planning", zorder=4)
    x, y, _ = sweep(pol, "indep", len(LAMS))
    f = front(x, y)
    ax.plot(x[f], y[f], "--s", ms=1.8, color=C["indep"], lw=0.9, label="Independent-step DP")
    ax.plot(x, y, "s", ms=1.8, color=C["indep"])
    per = sorted((pol[f"periodic-{T}"]["tokens"][0], pol[f"periodic-{T}"]["success"][0] * 100) for T in R["periods"])
    ax.plot([p[0] for p in per], [p[1] for p in per], ":^", ms=2.0, color=C["periodic"], lw=0.9, label="Periodic critic ($T$ sweep)")
    for key, mk, col, lab in (("every", "D", C["every"], "Critic every step (in place)"), ("every-fresh", "d", C["fresh"], "Critic every step (rollback)"),
                              ("vote3", "X", "#666666", "Vote of 3 per step"), ("none", "o", C["none"], "Single agent")):
        ax.plot(pol[key]["tokens"][0], pol[key]["success"][0] * 100, mk, ms=3.6, color=col, label=lab, zorder=6, mec="white", mew=0.3)
    ax.set_title(TITLE[env])
    ax.set_xlabel(r"Tokens ($\times$ single agent, log)")
    ax.set_xscale("log")
    ax.set_xlim(0.95, 3.4)
    ax.set_xticks([1, 1.5, 2, 3]); ax.set_xticklabels(["1", "1.5", "2", "3"]); ax.minorticks_off()
axs[0].set_ylabel("Task success (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.13), handlelength=1.6, columnspacing=0.9, handletextpad=0.4)
fig.tight_layout(w_pad=0.6)
fig.savefig(os.path.join(OUT, "fig_pareto.pdf"))
plt.close(fig)

# 2. does the model predict? ---------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.0), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
ax = axs[0]
marks = dict(retail="o", airline="s", tac="^", backoffice="D")
mae = {}
for kind, col in (("indep", C["indep"]), ("belay", C["belay"])):
    errs = []
    for env in ENVS:
        pol, pred = R["envs"][env]["policies"], R["envs"][env]["pred"]
        keys = [k.split(":", 1)[1] for k in pred if k.startswith(kind + ":")]
        xs = [pred[f"{kind}:{k}"]["success"] * 100 for k in keys]
        ys = [pol[k]["success"][0] * 100 for k in keys]
        errs += [a - b for a, b in zip(xs, ys)]
        ax.plot(xs, ys, marks[env], ms=2.6, color=col, alpha=0.85, mec="white", mew=0.25)
    mae[kind] = float(np.mean(np.abs(errs)))
ax.plot([0, 100], [0, 100], "-", color="#555555", lw=0.6, zorder=0)
ax.set_xlim(0, 80); ax.set_ylim(0, 80)
ax.set_xlabel("Predicted success (%)"); ax.set_ylabel("Measured success (%)")
ax.plot([], [], "o", color=C["belay"], ms=3, label=f"BELAY model (MAE {mae['belay']:.1f})")
ax.plot([], [], "o", color=C["indep"], ms=3, label=f"Independent steps (MAE {mae['indep']:.1f})")
ax.legend(loc="upper left", handletextpad=0.2)
ax.set_title("(a) Success of 13 policies, 4 benchmarks")

ax = axs[1]
pol, pred = R["envs"]["retail"]["policies"], R["envs"]["retail"]["pred"]
ks = np.arange(1, 9)
for key, col, lab in (("none", C["none"], "Single agent"), ("static-10", C["belay"], "BELAY plan")):
    m = np.array([p[0] for p in pol[key]["pass_k"]]) * 100
    ax.plot(ks, m, "-o", ms=2.2, color=col, label=f"{lab}, measured")
    p1 = pol[key]["success"][0]
    ax.plot(ks, 100 * p1 ** ks, ":", color=col, lw=0.9)
    ax.plot([4], [pred[f"belay:{key}"]["pass4"] * 100], "*", ms=5.5, color=col, mec="white", mew=0.3)
    ax.plot([4], [pred[f"indep:{key}"]["pass4"] * 100], "x", ms=4, color=C["indep"], mew=1.0)
ax.plot([], [], ":", color="#555555", label=r"(pass$^1)^k$")
ax.plot([], [], "*", color="#555555", ms=5, label=r"pass$^4$, BELAY model")
ax.plot([], [], "x", color=C["indep"], ms=4, label=r"pass$^4$, independent steps")
ax.set_xlabel(r"$k$"); ax.set_ylabel(r"pass$^k$ (%)")
ax.set_title(r"(b) $\tau$-bench retail, pass$^k$")
ax.legend(loc="upper right", handlelength=1.4, handletextpad=0.3)
ax.set_ylim(0, 75)

ax = axs[2]
rows = R["envs"]["backoffice"]["retry"]
names = [r["type"] for r in rows]
xx = np.arange(len(rows))
w = 0.2
for i, (key, col, lab) in enumerate((("first", "#bbbbbb", "First attempt"), ("selection_only", C["static"], "Retry, selection only (model)"),
                                     ("fresh", C["belay"], "Retry, fresh context"), ("inplace", C["every"], "Retry, failed attempt in context"))):
    ax.bar(xx + (i - 1.5) * w, [100 * r[key] for r in rows], w, color=col, label=lab, lw=0)
ax.set_xticks(xx); ax.set_xticklabels(names)
ax.set_ylim(70, 101); ax.set_ylabel("Step success (%)")
ax.set_title("(c) Why retries fail (back-office)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, handlelength=1.0, columnspacing=0.6, handletextpad=0.3)
fig.tight_layout(w_pad=0.8)
fig.savefig(os.path.join(OUT, "fig_model.pdf"))
plt.close(fig)

# 3. horizon --------------------------------------------------------------------------------
H = R["horizon"]
ns = sorted({int(k.split("|")[0]) for k in H if k.endswith("|1.0")})
fig, axs = plt.subplots(1, 2, figsize=(COL, 1.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
ax = axs[0]
for pol, col, mk, lab in (("belay", C["belay"], "o", "BELAY"), ("indep", C["indep"], "s", "Indep. DP"),
                          ("periodic", C["periodic"], "^", "Periodic"), ("none", C["none"], ".", "None"), ("every", C["every"], "D", r"Every step (2.1$\times$)")):
    y = np.array([H[f"{n}|1.0"][pol]["success"][0] for n in ns]) * 100
    ax.plot(ns, y, "--" if pol == "every" else "-", marker=mk, ms=2.2, color=col, lw=1.2 if pol == "belay" else 0.8, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_yticks([2, 5, 10, 20, 50, 100]); ax.set_yticklabels(["2", "5", "10", "20", "50", "100"]); ax.minorticks_off()
ax.set_xlabel("Steps per job $n$"); ax.set_ylabel("Success (%, log)")
ax.set_title(r"(a) At $\leq$1.5$\times$ tokens", fontsize=7.5)
ax.legend(loc="lower left", handlelength=1.3, handletextpad=0.3, fontsize=5.5)
ax = axs[1]
hl = R["half_life"]
order = [("none", "None", C["none"]), ("periodic", "Periodic", C["periodic"]), ("indep", "Indep. DP", C["indep"]),
         ("belay-nf", "BELAY,\nno frailty", C["static"]), ("belay", "BELAY", C["belay"]), ("every", "Every step\n(2.1$\\times$)", C["every"])]
vals = [hl[k] or 0 for k, _, _ in order]
ax.barh(range(len(order)), vals, color=[c for _, _, c in order], height=0.65, lw=0)
for i, v in enumerate(vals):
    ax.text(v + 4, i, f"{v:.0f}", va="center", fontsize=6)
ax.set_yticks(range(len(order))); ax.set_yticklabels([l for _, l, _ in order], fontsize=5.8)
ax.invert_yaxis(); ax.set_xlabel("Half-life (steps)"); ax.set_xlim(0, max(vals) * 1.25)
ax.set_title("(b) Length at 50% success", fontsize=7.5)
ax.grid(axis="y", visible=False)
fig.tight_layout(w_pad=0.4)
fig.savefig(os.path.join(OUT, "fig_horizon.pdf"))
plt.close(fig)

# 4. where checks go ------------------------------------------------------------------------
P = R["placement"]
types = np.array(P["types"])
n = len(types)
fig, ax = plt.subplots(figsize=(COL, 1.75))
show = [LAMS[2], LAMS[6], LAMS[8], R["lam_star"], LAMS[-1]]
rows = [r for lam in show for r in P["rows"] if r["lam"] == lam and r["d"] == 0]
shapes = {0: "|", 1: "s", 2: "o", 3: "*"}
cname = P["check_names"]
cols = ["#009E73", "#E69F00", C["belay"], "#CC79A7"]
ylabels = []
for r_i, r in enumerate(rows):
    yv = len(rows) - r_i
    ylabels.append((yv, f"$\\lambda_0$={r['lam']}"))
    ax.plot([0.5, n + 0.5], [yv, yv], "-", color="#dddddd", lw=0.4, zorder=0)
    for b, v in r["checks"]:
        ax.plot(b, yv, marker="o" if cname[v] == "critic" else "|", ms=3.0 if cname[v] == "critic" else 5, color=cols[v], mew=1.0, zorder=3)
irr = np.flatnonzero(np.array(P["irreversible"])) + 1
for b in irr:
    ax.axvline(b, color="#cccccc", lw=0.5, ls=":", zorder=0)
ax.set_yticks([y for y, _ in ylabels]); ax.set_yticklabels([l for _, l in ylabels], fontsize=5.5)
ax.set_xlabel("Step of one 60-step back-office job (dotted: irreversible)")
ax.set_xlim(0, n + 1); ax.grid(False)
for v, nm in enumerate(cname):
    ax.plot([], [], marker="o" if nm == "critic" else "|", ls="", color=cols[v], label=nm)
ax.legend(loc="upper center", ncol=len(cname), bbox_to_anchor=(0.5, 1.2), handletextpad=0.2, columnspacing=0.8)
fig.savefig(os.path.join(OUT, "fig_placement.pdf"))
plt.close(fig)

# 5. break-even recall (appendix) -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(COL, 2.0))
ps = np.linspace(0.80, 0.995, 200)
qs = np.linspace(0.4, 1.0, 200)
PP, QQ = np.meshgrid(ps, qs)
f = 0.07
rstar = PP * f * (1 - QQ) / ((1 - PP) * QQ)
im = ax.contourf(PP * 100, QQ * 100, np.clip(rstar, 0, 1.2), levels=np.linspace(0, 1.2, 13), cmap="Blues")
ax.contour(PP * 100, QQ * 100, rstar, levels=[1.0], colors=["#D55E00"], linewidths=1.0)
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("Break-even recall", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
fits = R["envs"]["backoffice"]["fits"][0]
crit = [c for c in fits["checks"] if c["name"] == "critic"][0]
for k, name in enumerate(R["envs"]["backoffice"]["retry"] and [r["type"] for r in R["envs"]["backoffice"]["retry"]] or []):
    row = R["envs"]["backoffice"]["retry"][k]
    ax.plot(100 * row["first"], 100 * row["inplace"], "o", ms=3, color="#D55E00", mec="white", mew=0.3)
    ax.plot(100 * row["first"], 100 * row["fresh"], "o", ms=3, color="#0072B2", mec="white", mew=0.3)
    ax.annotate(name, (100 * row["first"], 100 * row["fresh"]), xytext=(2, 2), textcoords="offset points", fontsize=5.5)
ax.set_xlabel("Step success $p$ (%)"); ax.set_ylabel("Retry success $q$ (%)")
ax.set_title(f"False-flag rate {f:.2f}; right of the red line no recall helps", fontsize=6.5)
ax.grid(False)
fig.savefig(os.path.join(OUT, "fig_breakeven.pdf"))
plt.close(fig)

# 6. transfer (appendix) --------------------------------------------------------------------
T = R["transfer"]
fig, ax = plt.subplots(figsize=(COL, 1.8))
for shift, col, lab in (("airline", C["indep"], "Retail $\\to$ airline"), ("stronger", C["belay"], "Stronger agent"), ("weaker", C["every"], "Weaker agent")):
    nn = sorted({int(k.split("|")[1]) for k in T if k.startswith(shift + "|")})
    y = np.array([T[f"{shift}|{m}"]["success"][0] for m in nn]) * 100
    e = np.array([T[f"{shift}|{m}"]["success"][1] for m in nn]) * 100
    ax.errorbar(nn, y, yerr=e, fmt="-o", ms=2.4, color=col, lw=1.0, capsize=1.5, elinewidth=0.6, label=lab)
ax.set_xlabel("Target tasks used to refit (0: source plan)"); ax.set_ylabel("Success (%)")
ax.legend(loc="lower right", handlelength=1.4)
fig.savefig(os.path.join(OUT, "fig_transfer.pdf"))
plt.close(fig)
print("figures written to", OUT)
