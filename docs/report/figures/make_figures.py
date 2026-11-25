"""Regenerate every figure used in docs/report/REPORT.md and the slide deck.

Run from the repo root:

    python docs/report/figures/make_figures.py

Reads committed files only (results/**, reports/*/data.json, data/processed/*)
and writes PNG and SVG files next to this script. Nothing is simulated and no
random numbers are drawn, so two runs give the same files.

Seeds are always shown one by one. No figure averages the seeds' portfolios.
Units are certainty-equivalent bps per year unless an axis says otherwise.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ANN = 252 * 1e4  # daily utility -> bps per year

# One palette for the whole report.
NAVY = "#1d3557"    # continuous optimum, oracle, reference lines
BLUE = "#457b9d"    # best grid portfolio, second series
TEAL = "#2a9d8f"    # the learned policy / agent
ORANGE = "#e76f51"  # learning error, losses
GOLD = "#e9c46a"    # third series
GREY = "#8d99ae"    # equal weight, neutral references
LIGHT = "#d9dee7"
INK = "#22303f"
LEVELS = ("zero", "aum_100m", "aum_1b", "aum_10b")
LEVEL_LABEL = {"zero": "zero cost", "aum_100m": "$100M", "aum_1b": "$1B", "aum_10b": "$10B"}
LEVEL_COLOR = {"zero": GREY, "aum_100m": BLUE, "aum_1b": NAVY, "aum_10b": ORANGE}

plt.rcParams.update({
    "font.size": 12, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#5b6777", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 10.5,
    "figure.dpi": 100, "savefig.dpi": 200,
    "svg.hashsalt": "report-figures", "svg.fonttype": "none",
})


# ---------------------------------------------------------------- loading
def jload(rel):
    return json.loads((REPO / rel).read_text())


def rows(rel):
    with (REPO / rel).open() as fh:
        return list(csv.DictReader(fh))


_cache: dict = {}


def report(run_id):
    if run_id not in _cache:
        _cache[run_id] = jload(f"reports/{run_id}/data.json")
    return _cache[run_id]


def experiment(run_id, family):
    for e in report(run_id)["experiments"]:
        if e["family"] == family:
            return e
    raise KeyError((run_id, family))


def seed_gaps(run_id, family):
    return np.array([s["loss_best"] for s in experiment(run_id, family)["seeds"]])


def tc(level):
    key = ("tc", level)
    if key not in _cache:
        folder = f"results/tc-drift/runs/{level}"
        _cache[key] = {"summary": jload(f"{folder}/summary.json"), "rows": rows(f"{folder}/per_seed.csv")}
    return _cache[key]


def col(rs, key):
    return np.array([float(r[key]) if r[key] not in ("", "None") else np.nan for r in rs])


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", metadata={"Date": None} if ext == "svg" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def spread(n, width=0.3):
    return np.linspace(-width / 2, width / 2, n)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- figures
def fig_gap_ladder():
    """N7. The four portfolios on one certainty-equivalent axis (gamma 5, 20% grid)."""
    s = jload("results/baseline_g5_grid20/evaluation/summary.json")
    b = s["benchmarks"]
    u = {k: b[k]["exact"]["objective_daily"] * ANN for k in ("continuous", "best_discrete", "equal_weight")}
    learned = u["continuous"] - s["learned"]["ce_loss_vs_continuous_bps"]["median"]
    names = ["Equal weight", "Learned (median across seeds)", "Best grid portfolio", "Continuous optimum"]
    vals = [u["equal_weight"], learned, u["best_discrete"], u["continuous"]]
    cols = [GREY, TEAL, BLUE, NAVY]
    fig, ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    ax.barh(names, vals, color=cols, height=0.62)
    for i, v in enumerate(vals):
        gap = u["continuous"] - v
        txt = f"{v:.1f}" + (f"   ({gap:.1f} behind the optimum)" if gap > 0.05 else "")
        ax.text(v + 6, i, txt, va="center", fontsize=11.5)
    ax.set_xlim(0, 800)
    ax.set_xlabel("Certainty-equivalent return, bps/yr (γ = 5)")
    grid(ax, "x")
    save(fig, "fig_gap_ladder")


def fig_correlation():
    """N1. Correlation matrix of the nine ETFs' daily excess returns."""
    rs = rows("data/processed/correlation_matrix.csv")
    tick = [r[""] for r in rs]
    m = np.array([[float(r[t]) for t in tick] for r in rs])
    fig, ax = plt.subplots(figsize=(6.4, 5.4), layout="constrained")
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(9), tick)
    ax.set_yticks(range(9), tick)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=9.5,
                    color="white" if abs(m[i, j]) > 0.6 else INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation of daily excess returns")
    save(fig, "fig_correlation")


def fig_benchmarks_by_gamma():
    """N3. Continuous long-only optimum and best 20% grid portfolio at gamma 3, 5, 8."""
    folders = {3: "robust_g3_grid20", 5: "baseline_g5_grid20", 8: "robust_g8_grid20"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), layout="constrained", sharey=True)
    for ax, (g, folder) in zip(axes, folders.items()):
        w = {r["ticker"]: r for r in rows(f"results/{folder}/evaluation/weights.csv")}
        names = ["SPY", "IEF", "GLD"]
        x = np.arange(3)
        cont = [float(w[n]["continuous"]) for n in names]
        best = [float(w[n]["best_discrete"]) for n in names]
        ax.bar(x - 0.2, cont, 0.38, color=NAVY, label="Continuous optimum w*")
        ax.bar(x + 0.2, best, 0.38, color=BLUE, label="Best 20% grid portfolio")
        for xi, c, d in zip(x, cont, best):
            ax.text(xi - 0.2, c + 0.012, f"{c:.2f}", ha="center", fontsize=10)
            ax.text(xi + 0.2, d + 0.012, f"{d:.1f}", ha="center", fontsize=10)
        ax.set_xticks(x, names)
        ax.set_title(f"γ = {g}")
        ax.set_ylim(0, 0.72)
        grid(ax)
    axes[0].set_ylabel("Portfolio weight (other six ETFs are 0)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, fontsize=9.5)
    save(fig, "fig_benchmarks_by_gamma")


def fig_epsilon():
    """N2. Exploration schedule eps_t = exp(-beta t) of the pre-registered runs."""
    p = experiment("baseline-v1", "gamma5_grid20")["params"]
    n = p["n_episodes"]
    t = np.linspace(0, n, 600)
    eps = p["epsilon0"] * np.exp(-p["beta"] * t)
    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    ax.plot(t / 1e6, eps, color=TEAL, linewidth=2.4)
    ax.set_yscale("log")
    ax.axvline(n / 2e6, color=GREY, linestyle="--", linewidth=1)
    ax.scatter([n / 2e6], [p["epsilon_at_half"]], color=ORANGE, zorder=5, s=50)
    ax.annotate("ε = 1% at the halfway episode", (n / 2e6, p["epsilon_at_half"]), xytext=(7.0, 0.08),
                arrowprops={"arrowstyle": "-", "color": GREY}, fontsize=11)
    ax.set_xlabel("Training episode (millions)")
    ax.set_ylabel("Exploration rate ε (log scale)")
    ax.set_yticks([1, 0.1, 0.01, 0.001, 0.0001], ["100%", "10%", "1%", "0.1%", "0.01%"])
    grid(ax)
    save(fig, "fig_epsilon")


def fig_weights():
    """F1 rebuilt. Benchmarks as grouped bars; each seed's final portfolio in its own row."""
    w = rows("results/baseline_g5_grid20/evaluation/weights.csv")
    tick = [r["ticker"] for r in w]
    e = experiment("baseline-v1", "gamma5_grid20")
    seeds = sorted(e["seeds"], key=lambda s: (s["loss_best"], s["run"]))
    median_seed = seeds[5]  # the 6th of 10 by gap, as in the committed evaluation chart
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), layout="constrained", width_ratios=[1.25, 1])
    x = np.arange(9)
    series = [("Continuous optimum w*", [float(r["continuous"]) for r in w], NAVY),
              ("Best 20% grid portfolio", [float(r["best_discrete"]) for r in w], BLUE),
              (f"Learned, median seed (seed {median_seed['run']})", median_seed["weights"], TEAL),
              ("Equal weight", [float(r["equal_weight"]) for r in w], GREY)]
    for k, (lab, v, c) in enumerate(series):
        a.bar(x + (k - 1.5) * 0.2, v, 0.2, color=c, label=lab)
    a.set_xticks(x, tick)
    a.set_ylabel("Portfolio weight")
    a.set_title("Benchmarks and the median seed")
    a.legend(loc="upper center", fontsize=9.5)
    a.set_ylim(0, 0.72)
    grid(a)

    order = sorted(e["seeds"], key=lambda s: (s["loss_best"], s["run"]))
    m = np.array([s["weights"] for s in order])
    b.imshow(m, cmap="GnBu", vmin=0, vmax=0.75, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(9):
            if m[i, j] > 0:
                b.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=10,
                       color="white" if m[i, j] >= 0.6 else INK)
    b.set_xticks(range(9), tick)
    b.set_yticks(range(10), [f"seed {s['run']}   {s['loss_best']:.1f}" for s in order], fontsize=10)
    b.set_title("Each seed's final portfolio (label: learning gap, bps/yr)")
    b.tick_params(length=0)
    for sp in b.spines.values():
        sp.set_visible(False)
    b.set_xticks(np.arange(-0.5, 9), minor=True)
    b.set_yticks(np.arange(-0.5, 10), minor=True)
    b.grid(which="minor", color="white", linewidth=1.5)
    b.tick_params(which="minor", length=0)
    save(fig, "fig_weights")


BASE_FAMILIES = [("gamma5_grid20", "γ = 5\n20% grid"), ("gamma5_grid25", "γ = 5\n25% grid"),
                 ("gamma3_grid20", "γ = 3\n20% grid"), ("gamma8_grid20", "γ = 8\n20% grid")]


def fig_gaps():
    """F18 and F19 rebuilt. Gap decomposition and every seed's total gap, four pre-registered experiments."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4), layout="constrained")
    x = np.arange(4)
    disc = [experiment("baseline-v1", f)["summary"]["disc_gap_bps"] for f, _ in BASE_FAMILIES]
    med = [experiment("baseline-v1", f)["summary"]["learning_gap"]["median"] for f, _ in BASE_FAMILIES]
    a.bar(x, disc, 0.55, color=BLUE, label="Discretization gap (best grid vs. w*)")
    a.bar(x, med, 0.55, bottom=disc, color=ORANGE, label="Learning gap, median across seeds")
    for i in range(4):
        a.text(i, disc[i] / 2, f"{disc[i]:.1f}", ha="center", va="center", color="white", fontsize=11)
        if med[i] > 1:
            a.text(i, disc[i] + med[i] / 2, f"{med[i]:.1f}", ha="center", va="center", color="white", fontsize=11)
        a.text(i, disc[i] + med[i] + 0.6, f"{disc[i] + med[i]:.1f}", ha="center", fontsize=11, fontweight="bold")
    a.set_xticks(x, [lab for _, lab in BASE_FAMILIES])
    a.set_ylabel("Loss against the continuous optimum, bps/yr")
    a.set_title("Where the median shortfall across seeds comes from")
    a.set_ylim(0, 27)
    a.legend(loc="upper right", fontsize=9.5)
    grid(a)

    for i, (f, _) in enumerate(BASE_FAMILIES):
        e = experiment("baseline-v1", f)
        tot = np.sort([s["loss_cont"] for s in e["seeds"]])
        b.scatter(i + spread(len(tot), 0.42), tot, s=46, color=TEAL, alpha=0.85, zorder=3,
                  label="One seed" if i == 0 else None)
        b.hlines(disc[i], i - 0.32, i + 0.32, color=BLUE, linewidth=2.2, zorder=2,
                 label="Floor: the best grid portfolio" if i == 0 else None)
    b.set_xticks(x, [lab for _, lab in BASE_FAMILIES])
    b.set_ylabel("Total gap to the continuous optimum, bps/yr")
    b.set_title("Every seed's total gap")
    b.set_ylim(0, 90)
    b.legend(loc="upper left", fontsize=9.5)
    grid(b)
    save(fig, "fig_gaps")


def fig_learning_curve():
    """F3/F4/F10/F11 rebuilt. Moving-average reward and running average regret, 20% and 25% grids."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.3), layout="constrained")
    for fam, lab, c in (("gamma5_grid20", "20% grid (1,287 portfolios)", TEAL),
                        ("gamma5_grid25", "25% grid (495 portfolios)", GOLD)):
        e = experiment("baseline-v1", fam)
        xs = np.array(e["series"]["x_episodes"]) / 1e6
        a.plot(xs, np.array(e["series"]["reward_median"]) * 100, color=c, linewidth=2, label=lab)
        b.plot(xs, e["series"]["regret_median"], color=c, linewidth=2, label=lab)
    e = experiment("baseline-v1", "gamma5_grid20")
    a.axhline(e["benchmarks"]["u_cont_daily"] * ANN, color=NAVY, linestyle="--", linewidth=1.3,
              label="Continuous optimum")
    a.axhline(e["benchmarks"]["u_best_daily"] * ANN, color=BLUE, linestyle=":", linewidth=1.6,
              label="Best 20% grid portfolio")
    a.set_ylim(0, 600)
    a.set_xlabel("Training episode (millions; 20 simulated days each)")
    a.set_ylabel("Reward as certainty equivalent, bps/yr")
    a.set_title("Learning curve: moving-average training reward")
    a.legend(loc="lower right", fontsize=9.5)
    grid(a)
    b.set_xlabel("Training episode (millions)")
    b.set_ylabel("Average regret = cumulative regret ÷ episodes so far,\nbps/yr per episode")
    b.set_title("Average regret against the best grid portfolio")
    b.set_ylim(0, 500)
    b.legend(loc="upper right", fontsize=9.5)
    grid(b)
    save(fig, "fig_learning_curve")


def _loss_band(ax, e, color, label, share=False):
    xs = np.array(e["series"]["x_episodes"], dtype=float)
    xs = xs / xs[-1] * 100 if share else xs / 1e6
    ax.fill_between(xs, e["series"]["loss_q1"], e["series"]["loss_q3"], color=color, alpha=0.18, linewidth=0)
    ax.plot(xs, e["series"]["loss_median"], color=color, linewidth=2, label=label)


def fig_convergence_resolution():
    """F5/F6/F9 rebuilt. Greedy portfolio's loss over training at two pairs of resolutions (gamma 5)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4), layout="constrained", sharey=True)
    _loss_band(a, experiment("baseline-v1", "gamma5_grid20"), TEAL, "20% grid (1,287)")
    _loss_band(a, experiment("baseline-v1", "gamma5_grid25"), GOLD, "25% grid (495)")
    a.set_title("Pre-registered: 20% against 25% grid")
    a.set_xlabel("Training episode (millions; B = 20 days)")
    _loss_band(b, experiment("all-three", "gamma5_grid20"), TEAL, "20% grid (1,287)")
    _loss_band(b, experiment("grid10-robustness", "gamma5_grid10"), ORANGE, "10% grid (43,758)")
    b.set_title("Post hoc: 20% against 10% grid (B = 63, 26M episodes)")
    b.set_xlabel("Training episode (millions; B = 63 days)")
    for ax in (a, b):
        ax.set_yscale("symlog", linthresh=10, linscale=0.8)
        ax.set_ylim(0, 1000)
        ax.set_yticks([0, 5, 10, 25, 100, 1000], ["0", "5", "10", "25", "100", "1,000"])
        ax.axhline(25, color=GREY, linestyle=":", linewidth=1.2)
        ax.legend(loc="upper right", fontsize=10)
        grid(ax)
    b.text(25.8, 27, "25 bps/yr near-optimal line", fontsize=9.5, color="#5b6777", va="bottom", ha="right")
    a.set_ylabel("Greedy portfolio's loss vs. its grid's best, bps/yr\n(median across seeds; band = middle half)")
    save(fig, "fig_convergence_resolution")


def fig_noise():
    """N5 and F16 rebuilt. Reward noise against the gaps, and tries of the best portfolio."""
    d = jload("reports/full_run_2026-10-22/data.json")
    cal = d["calibration"]
    g20 = d["grids"]["g20"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.3), layout="constrained", width_ratios=[1, 1.15])
    names = ["One episode's reward noise (SD)", "Equal weight vs. best grid",
             "Best vs. 2nd-best grid portfolio", "Best grid vs. continuous optimum"]
    eq = (g20["u_best"] - jload("results/baseline_g5_grid20/evaluation/summary.json")
          ["benchmarks"]["equal_weight"]["exact"]["objective_daily"]) * ANN
    vals = [cal["reward_sd_best_daily"] * ANN, eq, cal["gap_best_second_bps"], g20["disc_gap_bps"]]
    a.barh(names[::-1], vals[::-1], color=[BLUE, ORANGE, GREY, NAVY], height=0.6)  # optimum gap, top-two gap, equal weight, noise
    for i, v in enumerate(vals[::-1]):
        a.text(v * 1.15, i, f"{v:,.1f}" if v < 100 else f"{v:,.0f}", va="center", fontsize=11.5)
    a.set_xscale("log")
    a.set_xlim(1, 40000)
    a.set_xlabel("bps/yr (log scale)")
    a.set_title("Signal and noise at γ = 5, B = 20")
    grid(a, "x")

    k = 0
    for gkey, lab, c in (("g20", "20% grid", TEAL), ("g25", "25% grid", GOLD)):
        g = d["grids"][gkey]
        missed = sorted(s["visits_best"] for s in g["seeds"] if not s["is_best"])
        xs = k + spread(len(missed), 0.5)
        b.scatter(xs, missed, s=52, color=c, zorder=3, label=f"{lab}: seeds that missed the best portfolio")
        b.hlines(g["visits_each_to_separate_2se"], k - 0.4, k + 0.4, color=NAVY, linewidth=2.2,
                 label="Tries needed to separate the top two (2 SE)" if k == 0 else None)
        b.text(k + 0.42, g["visits_each_to_separate_2se"], f"{g['visits_each_to_separate_2se'] / 1e6:.1f}M",
               va="center", fontsize=11)
        k += 1
    b.set_yscale("log")
    b.set_ylim(300, 3e7)
    b.set_xlim(-0.6, 1.9)
    b.set_xticks([0, 1], ["20% grid", "25% grid"])
    b.set_ylabel("Times the best portfolio was tried (log)")
    b.set_title("The best portfolio was tried too rarely")
    b.legend(loc="center left", fontsize=9, bbox_to_anchor=(0.0, 0.5))
    grid(b)
    save(fig, "fig_noise")


def fig_convergence_times():
    """F12 rebuilt. When each convergence measure is reached, one dot per seed."""
    measures = [("settled_obs", "Settled on final portfolio\n(post hoc)"),
                ("near_obs_ckpt", "Within 25 bps/yr for good\n(economic)"),
                ("dq_obs", "Largest Q change below 1e-5"),
                ("first_stable_obs", "Greedy unchanged for K episodes"),
                ("primary_obs", "Primary rule: unchanged\nthrough the end")]
    fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
    for k, (fam, lab, c) in enumerate((("gamma5_grid20", "20% grid", TEAL), ("gamma5_grid25", "25% grid", GOLD))):
        e = experiment("baseline-v1", fam)
        for i, (key, _) in enumerate(measures):
            v = np.array(sorted(s[key] for s in e["seeds"] if s[key] is not None)) / 1e6
            y = i + (0.17 if k == 0 else -0.17)
            ax.scatter(v, np.full(len(v), y), s=40, color=c, alpha=0.85, zorder=3,
                       label=lab if i == 0 else None, edgecolor="white", linewidth=0.4)
            ax.text(266, y, f"{len(v)} of 10", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(measures)), [m for _, m in measures])
    ax.set_xlim(0, 262)
    ax.set_xlabel("Simulated days of training when the measure is first met (millions; budget 260M)")
    ax.invert_yaxis()
    ax.legend(loc="lower left", ncols=2, bbox_to_anchor=(0, 1.0))
    ax.text(266, -0.62, "seeds meeting it", fontsize=9.5, color="#5b6777")
    grid(ax, "x")
    save(fig, "fig_convergence_times")


def fig_seed_paths():
    """F14 rebuilt. Each seed's greedy portfolio loss over training (gamma 5, 20% grid)."""
    e = experiment("baseline-v1", "gamma5_grid20")
    xs = np.array(e["series"]["x_episodes"]) * e["params"]["episode_length"] / 1e6
    fig, axes = plt.subplots(2, 5, figsize=(13, 5), layout="constrained", sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), e["seeds"]):
        ax.plot(xs, s["series"]["loss"], color=TEAL, linewidth=1.1)
        ax.axvline(s["settled_obs"] / 1e6, color=GOLD, linewidth=1.6)
        if s["primary_obs"] is not None:
            ax.axvline(s["primary_obs"] / 1e6, color=NAVY, linewidth=1.6, linestyle="--")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(0, 2000)
        ax.set_yticks([0, 10, 100, 1000], ["0", "10", "100", "1,000"])
        tag = "best grid portfolio" if s["is_best"] else f"gap {s['loss_best']:.1f}"
        ax.set_title(f"seed {s['run']}: {tag}", fontsize=10.5, fontweight="normal")
        grid(ax)
    fig.supxlabel("Simulated days of training (millions).  Gold line: settled.  Dashed navy: primary rule met.",
                  fontsize=11)
    fig.supylabel("Greedy portfolio's loss vs. best grid, bps/yr", fontsize=11)
    save(fig, "fig_seed_paths")


def fig_grid10():
    """F6 rebuilt for all three gammas: 20% against 10% grid at the all-three settings."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), layout="constrained", sharey=True)
    for ax, g in zip(axes, (5, 3, 8)):
        _loss_band(ax, experiment("all-three", f"gamma{g}_grid20"), TEAL, "20% grid (1,287)")
        _loss_band(ax, experiment("grid10-robustness", f"gamma{g}_grid10"), ORANGE, "10% grid (43,758)")
        ax.set_yscale("symlog", linthresh=10, linscale=0.8)
        ax.set_ylim(0, 1000)
        ax.set_yticks([0, 5, 10, 25, 100, 1000], ["0", "5", "10", "25", "100", "1,000"])
        ax.axhline(25, color=GREY, linestyle=":", linewidth=1.2)
        ax.set_title(f"γ = {g}")
        ax.set_xlabel("Training episode (millions; B = 63)")
        grid(ax)
    axes[0].set_ylabel("Greedy loss vs. own grid's best, bps/yr\n(median across seeds; band = middle half)")
    axes[0].legend(loc="upper right", fontsize=10)
    save(fig, "fig_grid10")


VARIANTS = [("baseline-v1", "Baseline", GREY), ("b63", "B = 63", BLUE), ("n26m", "26M episodes", GOLD),
            ("eps-decay", "ε = 3% at half", ORANGE), ("all-three", "All three", TEAL)]


def _strip(ax, groups, runs, families, ymax):
    """Dots for every seed and a bar for the mean, by run within each gamma group."""
    n = len(runs)
    width = 0.8 / n
    for gi, fam in enumerate(families):
        for ri, (run_id, lab, c) in enumerate(runs):
            v = np.sort(seed_gaps(run_id, fam))
            x0 = gi + (ri - (n - 1) / 2) * width
            ax.bar(x0, v.mean(), width * 0.86, color=c, alpha=0.3, zorder=1)
            ax.scatter(x0 + spread(len(v), width * 0.6), np.minimum(v, ymax * 0.985), s=20, color=c, zorder=3,
                       label=lab if gi == 0 else None, edgecolor="white", linewidth=0.3)
            ax.hlines(v.mean(), x0 - width * 0.43, x0 + width * 0.43, color=INK, linewidth=1.6, zorder=4)
            ax.text(x0, min(v.mean(), ymax * 0.93) + ymax * 0.015, f"{v.mean():.1f}", ha="center", fontsize=8.5,
                    zorder=5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.75})
    ax.set_xticks(range(len(families)), groups)
    ax.set_ylim(0, ymax)
    grid(ax)


def fig_robustness():
    """N4. Learning gap by tabular variant and gamma: every seed and the mean."""
    fig, ax = plt.subplots(figsize=(11.5, 4.5), layout="constrained")
    _strip(ax, ["γ = 5", "γ = 3", "γ = 8"], VARIANTS, ["gamma5_grid20", "gamma3_grid20", "gamma8_grid20"], 90)
    ax.set_ylabel("Learning gap vs. best grid portfolio, bps/yr\n(dot = seed, bar and label = mean)")
    ax.legend(loc="upper center", ncols=5, bbox_to_anchor=(0.5, 1.13), fontsize=10)
    save(fig, "fig_robustness")


def fig_fa():
    """F23 and F24 rebuilt. Function approximation against the tables on both grids."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    runs20 = [("baseline-v1", "Table, α = 1/N", GREY), ("tabular-const", "Table, constant α", BLUE),
              ("fa-quadratic", "Quadratic features", GOLD), ("fa-spline", "Splines", TEAL)]
    _strip(a, ["γ = 5", "γ = 3", "γ = 8"], runs20, ["gamma5_grid20", "gamma3_grid20", "gamma8_grid20"], 215)
    a.set_title("20% grid (1,287 portfolios)", pad=30)
    a.set_ylabel("Learning gap, bps/yr\n(dot = seed, bar and label = mean)")
    a.legend(loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.115), fontsize=9.3, columnspacing=1,
             handletextpad=0.2)
    runs10 = [("grid10-robustness", "Table, α = 1/N", GREY), ("fa-spline-grid10-eps3", "Splines, ε 3% at half", TEAL),
              ("fa-spline-grid10-eps10", "Splines, ε 10% at half", ORANGE)]
    _strip(b, ["γ = 5", "γ = 3", "γ = 8"], runs10, ["gamma5_grid10", "gamma3_grid10", "gamma8_grid10"], 260)
    b.set_title("10% grid (43,758 portfolios)", pad=30)
    b.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.115), fontsize=9.3, columnspacing=1,
             handletextpad=0.2)
    save(fig, "fig_fa")


def fig_fa_ranking():
    """Overall ranking quality against quality at the top, per seed, 10% grid at gamma 5."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    for run_id, fam, lab, c, mk in (("baseline-v1", "gamma5_grid20", "Table 1/N, 20% grid", GREY, "o"),
                                    ("fa-spline", "gamma5_grid20", "Splines, 20% grid", BLUE, "o"),
                                    ("grid10-robustness", "gamma5_grid10", "Table 1/N, 10% grid", GOLD, "s"),
                                    ("fa-spline-grid10-eps3", "gamma5_grid10", "Splines, 10% grid", TEAL, "s")):
        e = experiment(run_id, fam)
        sp = np.array([s["q_spearman_all"] for s in e["seeds"]])
        gp = np.array([s["loss_best"] for s in e["seeds"]])
        ax.scatter(sp, gp, s=48, color=c, marker=mk, label=lab, alpha=0.9, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Rank correlation of learned Q with true utility over the whole grid (Spearman)")
    ax.set_ylabel("Learning gap of the chosen portfolio, bps/yr")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(-5, 150)
    ax.legend(loc="upper left", fontsize=10)
    grid(ax)
    grid(ax, "x")
    save(fig, "fig_fa_ranking")


def fig_tc_schematic():
    """N6. One 63-day block of the transaction-cost problem."""
    fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.95, 3.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, tc_="white", fs=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.1", fc=fc, ec="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc_, fontsize=fs)

    def arrow(x0, y0, x1, y1, text=None, dy=0.16):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, color=INK, linewidth=1.4))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", fontsize=9.8, color=INK)

    box(0.1, 1.3, 2.0, 1.0, "State: drifted\nholdings s", NAVY)
    box(3.5, 2.35, 2.9, 0.95, "TRADE to one of 1,326 targets\ncost κ‖trade‖₁ + (λ/2)‖trade‖₂²", ORANGE, fs=10.2)
    box(3.5, 0.3, 2.9, 0.95, "HOLD\ncost 0, keep s", TEAL)
    box(7.6, 1.3, 3.2, 1.0, "63 days: weights drift with returns\nreward = mean − (γ/2)·var − cost/63", BLUE, fs=10.2)
    arrow(2.15, 2.0, 3.45, 2.8)
    arrow(2.15, 1.6, 3.45, 0.8)
    arrow(6.45, 2.8, 7.55, 2.05)
    arrow(6.45, 0.8, 7.55, 1.55)
    ax.add_patch(FancyArrowPatch((9.2, 1.25), (1.1, 1.25), connectionstyle="arc3,rad=-0.42", arrowstyle="-|>",
                                 mutation_scale=16, color=GREY, linewidth=1.3, linestyle="--"))
    ax.text(5.2, -0.82, "end-of-block holdings become the next state (discount 0.95 per block)", ha="center",
            fontsize=9.8, color="#5b6777")
    save(fig, "fig_tc_schematic")


def fig_tc_first_design():
    """F36 rebuilt. First transaction-cost design: learned policy against its benchmarks."""
    d = jload("reports/tc-quadratic-cost/data.json")
    costs = [c["cost"] for c in d["costs"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    for key, lab, c, ls in (("grid_oracle", "Grid oracle", NAVY, "-"), ("ignore_cost", "Ignore costs", BLUE, "--"),
                            ("myopic", "Myopic", GOLD, ":")):
        ax.plot(costs, [c_["benchmarks"][key]["expected_net_ce_bps_per_year"] for c_ in d["costs"]], color=c,
                linestyle=ls, linewidth=2.2, marker="o", label=lab)
    for i, c_ in enumerate(d["costs"]):
        v = np.array([s["learned"]["expected_net_ce_bps_per_year"] for s in c_["seeds"]])
        ax.scatter(np.full(len(v), costs[i]) + spread(len(v), 0.3), v, s=30, color=ORANGE, zorder=3,
                   label="Learned policy, one dot per seed" if i == 0 else None)
    ax.axhline(0, color="#5b6777", linewidth=0.8)
    ax.set_xticks(costs, [f"{c} bp" for c in costs])
    ax.set_xlabel("Cost of a 20-point transfer between two ETFs")
    ax.set_ylabel("Expected net certainty equivalent, bps/yr")
    ax.legend(loc="lower left", fontsize=10, ncols=2, bbox_to_anchor=(0, 1.0))
    grid(ax)
    save(fig, "fig_tc_first_design")


def fig_tc_split():
    """F29 rebuilt. Where the agent's gap to the oracle comes from, by cost level."""
    x = np.arange(4)
    target = [tc(l)["summary"]["across_seeds"]["split_own_target_minus_always_rebalance_mean"]["mean"] for l in LEVELS]
    timing = [tc(l)["summary"]["across_seeds"]["split_agent_minus_own_target_mean"]["mean"] for l in LEVELS]
    lead = [-tc(l)["summary"]["oracle_minus_always_rebalance"]["mean"] for l in LEVELS]
    total = [tc(l)["summary"]["across_seeds"]["agent_minus_oracle_mean"]["mean"] for l in LEVELS]
    fig, ax = plt.subplots(figsize=(11.5, 4.5), layout="constrained")
    w = 0.2
    bars = [(target, ORANGE, "Where it heads: own target − always rebalance"),
            (timing, TEAL, "When it trades: agent − own target"),
            (lead, GREY, "Minus the oracle's lead over always rebalance")]
    for k, (v, c, lab) in enumerate(bars):
        xb = x + (k - 1.5) * w
        ax.bar(xb, v, w * 0.92, color=c, label=lab)
        for xi, vi in zip(xb, v):
            ax.text(xi, vi + (0.5 if vi >= 0 else -0.5), "0.0" if abs(vi) < 0.05 else f"{vi:+.1f}", ha="center",
                    va="bottom" if vi >= 0 else "top", fontsize=9.5)
    for i, l in enumerate(LEVELS):
        seeds = col(tc(l)["rows"], "agent_minus_oracle_mean")
        ax.scatter(np.full(len(seeds), i + 1.6 * w), seeds, s=18, color=NAVY, alpha=0.55, zorder=3,
                   label="Agent − oracle: each seed" if i == 0 else None)
        ax.hlines(total[i], i + 1.25 * w, i + 1.95 * w, color=NAVY, linewidth=2.6, zorder=4,
                  label="Agent − oracle: mean of 10 seeds" if i == 0 else None)
        ax.text(i + 2.05 * w, total[i], f"{total[i]:.1f}", va="center", fontsize=10, color=NAVY, fontweight="bold")
    ax.axhline(0, color="#5b6777", linewidth=0.9)
    ax.set_xticks(x, [LEVEL_LABEL[l] for l in LEVELS])
    ax.set_xlim(-0.55, 3.75)
    ax.set_ylim(-23, 11)
    ax.set_ylabel("bps/yr\n(the three bars sum to the mean)")
    ax.legend(loc="lower right", fontsize=9.3, ncols=2, bbox_to_anchor=(1.0, 1.0))
    grid(ax)
    save(fig, "fig_tc_split")


def fig_tc_targets_hold():
    """F31 and F32 rebuilt. Each seed's favourite target, and how often it does not trade."""
    best = tc("aum_1b")["summary"]["best_target"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4), layout="constrained")
    for k, l in enumerate(LEVELS):
        pts = np.array([[float(v) for v in r["favourite_target"].split()] for r in tc(l)["rows"]])
        j = (k - 1.5) * 0.003
        a.scatter(pts[:, 1] + j, pts[:, 0] + j, s=44, color=LEVEL_COLOR[l], alpha=0.85, label=LEVEL_LABEL[l],
                  edgecolor="white", linewidth=0.4)
    a.scatter(best[1], best[0], marker="*", s=380, color=GOLD, edgecolor=INK, linewidth=0.8, zorder=5,
              label="Best no-cost target")
    a.set_xlabel("IEF weight of the seed's favourite target")
    a.set_ylabel("SPY weight (GLD is the rest)")
    a.set_title("Where the agent heads")
    a.legend(loc="lower left", fontsize=9.5, ncols=5, bbox_to_anchor=(-0.02, 1.07), columnspacing=0.8,
             handletextpad=0.1)
    grid(a)
    grid(a, "x")
    for i, l in enumerate(LEVELS):
        v = np.sort(col(tc(l)["rows"], "agent_hold_fraction"))
        b.scatter(i + spread(len(v), 0.36), v, s=40, color=LEVEL_COLOR[l], edgecolor="white", linewidth=0.4,
                  label="Agent, one dot per seed" if i == 2 else None)
        b.scatter(i, tc(l)["summary"]["references"]["oracle"]["hold_fraction"], marker="D", s=70, color=GOLD,
                  edgecolor=INK, linewidth=0.8, zorder=4, label="Oracle" if i == 2 else None)
    b.set_xticks(range(4), [LEVEL_LABEL[l] for l in LEVELS])
    b.set_ylabel("Share of 63-day blocks with no trade")
    b.set_ylim(0, 0.8)
    b.set_title("How often it holds")
    b.legend(loc="lower left", fontsize=9.5, ncols=2, bbox_to_anchor=(-0.02, 1.07))
    grid(b)
    save(fig, "fig_tc_targets_hold")


def fig_tc_vs_references():
    """F30 rebuilt. Every seed against the oracle, always-rebalance and myopic policies."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), layout="constrained", sharey=True)
    for ax, (key, title) in zip(axes, (("agent_minus_oracle_mean", "Agent − oracle"),
                                       ("agent_minus_always_rebalance_mean", "Agent − always rebalance"),
                                       ("agent_minus_myopic_mean", "Agent − myopic"))):
        for i, l in enumerate(LEVELS):
            v = np.sort(col(tc(l)["rows"], key))
            ax.scatter(i + spread(len(v), 0.36), v, s=34, color=LEVEL_COLOR[l], alpha=0.9, edgecolor="white",
                       linewidth=0.4)
            ax.hlines(v.mean(), i - 0.3, i + 0.3, color=INK, linewidth=2)
            ax.text(i, 9.3, f"{v.mean():+.1f}", ha="center", fontsize=10)
        ax.axhline(0, color="#5b6777", linewidth=0.9)
        ax.set_xticks(range(4), [LEVEL_LABEL[l] for l in LEVELS])
        ax.set_title(title)
        ax.set_ylim(-23, 11.5)
        grid(ax)
    axes[0].set_ylabel("bps/yr (dot = seed, bar and label = mean)")
    save(fig, "fig_tc_vs_references")


def fig_tc_costs():
    """F33 in summary form. Final learned cost coefficients as a ratio to the truth, per seed."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.0), layout="constrained")
    lv = LEVELS[1:]
    for ax, key, title in ((a, "c1_over_true", "Linear coefficient (spread cost) / truth"),
                           (b, "c2_over_true", "Quadratic coefficient (impact cost) / truth")):
        for i, l in enumerate(lv):
            v = np.sort(col(tc(l)["rows"], key))
            ax.scatter(i + spread(len(v), 0.4), v, s=44, color=LEVEL_COLOR[l], edgecolor="white", linewidth=0.4,
                       zorder=3)
        ax.axhspan(0.5, 2, color=TEAL, alpha=0.12, linewidth=0)
        ax.axhline(1, color=INK, linestyle="--", linewidth=1)
        ax.set_yscale("symlog", linthresh=0.1, linscale=0.5)
        ax.set_ylim(0, 20)
        ax.set_yticks([0, 0.1, 0.5, 1, 2, 5, 10], ["0", "0.1", "0.5", "1", "2", "5", "10"])
        ax.set_xticks(range(3), [LEVEL_LABEL[l] for l in lv])
        ax.set_title(title)
        grid(ax)
    a.set_ylabel("Learned / true (band = within a factor of 2)")
    save(fig, "fig_tc_costs")


def fig_tc_values():
    """Table TC-1 as a chart: value of each policy by cost level, with every seed of the agent."""
    fig, ax = plt.subplots(figsize=(10.5, 4.4), layout="constrained")
    refs = [("oracle", "Oracle", NAVY, "D"), ("myopic", "Myopic", BLUE, "v"),
            ("always_rebalance", "Always rebalance", GREY, "s"), ("trade_once_then_hold", "Trade once, then hold", GOLD, "^")]
    for i, l in enumerate(LEVELS):
        s = tc(l)["summary"]
        for k, (key, lab, c, mk) in enumerate(refs):
            ax.scatter(i - 0.27 + 0.1 * k, s["references"][key]["mean_value_bps"], marker=mk, s=80, color=c,
                       edgecolor=INK, linewidth=0.6, zorder=4, label=lab if i == 0 else None)
        v = np.sort([p["agent"]["mean_value_bps"] for p in s["per_seed"]])
        ax.scatter(i + 0.22 + spread(len(v), 0.22), v, s=26, color=TEAL, zorder=3,
                   label="Agent, one dot per seed" if i == 0 else None)
    ax.set_xticks(range(4), [LEVEL_LABEL[l] for l in LEVELS])
    ax.set_ylabel("Value of the policy, bps/yr")
    ax.set_ylim(490, 536)
    ax.legend(loc="lower left", ncols=3, fontsize=9.5, bbox_to_anchor=(0, 1.0))
    grid(ax)
    save(fig, "fig_tc_values")


FIGURES = [fig_gap_ladder, fig_correlation, fig_benchmarks_by_gamma, fig_epsilon, fig_weights, fig_gaps,
           fig_learning_curve, fig_convergence_resolution, fig_noise, fig_convergence_times, fig_seed_paths,
           fig_grid10, fig_robustness, fig_fa, fig_fa_ranking, fig_tc_schematic, fig_tc_first_design,
           fig_tc_split, fig_tc_targets_hold, fig_tc_vs_references, fig_tc_costs, fig_tc_values]


if __name__ == "__main__":
    for f in FIGURES:
        f()
    print(f"{len(FIGURES)} figures written to {OUT}")
