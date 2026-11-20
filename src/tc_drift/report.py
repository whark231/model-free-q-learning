"""HTML report for the TC-with-drift final agent runs (Albert, 2026-11-20).

Built only from committed files: results/tc-drift/runs/<level>/summary.json,
per_seed.csv and seed<i>.npz, and results/tc-drift/oracle/summary.json. It
retrains and rescores nothing, so anyone can rebuild it from a clone:

    python -m src.tc_drift.report            # writes reports/tc-drift/report.html and data.json

The expectations E1 to E12 are the ones registered in TC_DRIFT_DESIGN.md §7
before the runs; `expectations()` applies each rule as written there.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.tc_drift import config, spec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "results/tc-drift/runs"
OUT = REPO / "reports/tc-drift"
LABEL = {"zero": "zero", "aum_100m": "$100M", "aum_1b": "$1B", "aum_10b": "$10B"}
COLOR = {"zero": "#8a94a6", "aum_100m": "#5b8db8", "aum_1b": "#12667c", "aum_10b": "#c0603a"}
COST_LEVELS = ("aum_100m", "aum_1b", "aum_10b")

plt.rcParams.update({"svg.hashsalt": "tc-drift-report", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def load(runs: Path = RUNS) -> dict:
    data = {}
    for lv in config.LEVELS:
        folder = runs / lv
        s = json.loads((folder / "summary.json").read_text())
        rows = list(csv.DictReader((folder / "per_seed.csv").open()))
        logs = {int(r["seed"]): np.load(folder / f"seed{r['seed']}.npz") for r in rows}
        data[lv] = {"summary": s, "rows": rows, "logs": logs}
    commits = {d["summary"]["provenance"]["commit"] for d in data.values()}
    dirty = {d["summary"]["provenance"]["src_dirty"] for d in data.values()}
    if len(commits) != 1 or dirty != {False}:
        raise ValueError(f"levels come from different or dirty code: {commits}, {dirty}")
    return data


def col(rows, key):
    return np.array([float(r[key]) if r[key] not in ("", "None") else np.nan for r in rows])


def mean_of(d, key):
    return d["summary"]["across_seeds"][key]["mean"]


def expectations(data: dict) -> list[dict]:
    """Each registered expectation (design §7), checked as written. Verdict is
    'met', 'partly met' or 'not met'; `parts` lists each clause."""
    out = []

    def add(tag, text, parts):
        ok = [p[1] for p in parts]
        verdict = "met" if all(ok) else ("not met" if not any(ok) else "partly met")
        out.append({"id": tag, "expectation": text, "verdict": verdict,
                    "parts": [{"clause": c, "met": bool(m), "observed": o} for c, m, o in parts]})

    lv = config.LEVELS
    g = {k: {l: mean_of(data[l], k) for l in lv} for k in (
        "agent_minus_oracle_mean", "agent_minus_always_rebalance_mean", "agent_minus_myopic_mean",
        "split_agent_minus_own_target_mean", "split_own_target_minus_always_rebalance_mean",
        "agent_hold_fraction", "agent_minus_last_iterate_mean")}
    neg = {l: int((col(data[l]["rows"], "agent_minus_oracle_mean") < 0).sum()) for l in lv}
    add("E1", "Agent − oracle between −1 and −8 at every level, negative on at least 8 of 10 seeds",
        [(f"mean in [−8, −1] at {LABEL[l]}", -8 <= g["agent_minus_oracle_mean"][l] <= -1,
          f"{g['agent_minus_oracle_mean'][l]:.2f}") for l in lv]
        + [(f"negative on ≥ 8 seeds at {LABEL[l]}", neg[l] >= 8, f"{neg[l]}/10") for l in lv])
    add("E2", "Favourite-target error is most of the gap at zero, $100M, $1B",
        [(f"own target − AR < 0 and larger than |agent − own target| at {LABEL[l]}",
          g["split_own_target_minus_always_rebalance_mean"][l] < 0
          and abs(g["split_own_target_minus_always_rebalance_mean"][l]) > abs(g["split_agent_minus_own_target_mean"][l]),
          f"{g['split_own_target_minus_always_rebalance_mean'][l]:.2f} vs {g['split_agent_minus_own_target_mean'][l]:.2f}")
         for l in lv[:3]])
    ar = g["agent_minus_always_rebalance_mean"]
    add("E3", "Agent loses to always rebalance at $100M and $1B; relatively better at $10B than $1B",
        [("agent − AR < 0 at $100M", ar["aum_100m"] < 0, f"{ar['aum_100m']:.2f}"),
         ("agent − AR < 0 at $1B", ar["aum_1b"] < 0, f"{ar['aum_1b']:.2f}"),
         ("$10B above $1B", ar["aum_10b"] > ar["aum_1b"], f"{ar['aum_10b']:.2f} vs {ar['aum_1b']:.2f}")])
    t = g["split_agent_minus_own_target_mean"]
    add("E4", "Agent − own target rises from $1B to $10B and is positive at $10B; within ±1 at zero and $100M",
        [("rises $1B → $10B", t["aum_10b"] > t["aum_1b"], f"{t['aum_1b']:.2f} → {t['aum_10b']:.2f}"),
         ("positive at $10B", t["aum_10b"] > 0, f"{t['aum_10b']:.2f}"),
         ("within ±1 at zero", abs(t["zero"]) <= 1, f"{t['zero']:.2f}"),
         ("within ±1 at $100M", abs(t["aum_100m"]) <= 1, f"{t['aum_100m']:.2f}")])
    add("E5", "Agent beats trade once, then hold at every level and on every seed",
        [(f"every seed > 0 at {LABEL[l]}", (col(data[l]["rows"], "agent_minus_trade_once_then_hold_mean") > 0).all(),
          f"lowest {col(data[l]['rows'], 'agent_minus_trade_once_then_hold_mean').min():.2f}") for l in lv])
    add("E6", "Agent loses to the myopic oracle at every level",
        [(f"agent − myopic < 0 at {LABEL[l]}", g["agent_minus_myopic_mean"][l] < 0,
          f"{g['agent_minus_myopic_mean'][l]:.2f}") for l in lv])
    h = g["agent_hold_fraction"]
    oh = data["zero"]["summary"]["references"]["oracle"]["hold_fraction"]
    spread = {l: col(data[l]["rows"], "agent_hold_fraction") for l in lv}
    add("E7", "Hold fraction varies widely across seeds; higher at $10B than zero; at zero above the oracle's",
        [("wide spread (max − min ≥ 0.3) at the headline $1B", np.ptp(spread["aum_1b"]) >= 0.3,
          f"{spread['aum_1b'].min():.2f} to {spread['aum_1b'].max():.2f}"),
         ("mean higher at $10B than at zero", h["aum_10b"] > h["zero"], f"{h['aum_10b']:.3f} vs {h['zero']:.3f}"),
         ("at zero above the oracle's", h["zero"] > oh, f"{h['zero']:.3f} vs {oh:.3f}")])
    within2 = lambda x: (x >= 0.5) & (x <= 2)  # noqa: E731
    parts = [(f"c2 within ×2 on ≥ 7 seeds at {LABEL[l]}", within2(col(data[l]["rows"], "c2_over_true")).sum() >= 7,
              f"{int(within2(col(data[l]['rows'], 'c2_over_true')).sum())}/10") for l in ("aum_1b", "aum_10b")]
    parts += [(f"c1 off by > ×2 on ≥ 3 seeds at {LABEL[l]}", (~within2(col(data[l]["rows"], "c1_over_true"))).sum() >= 3,
               f"{int((~within2(col(data[l]['rows'], 'c1_over_true'))).sum())}/10") for l in COST_LEVELS]
    add("E8", "c2 learned better than c1 (benchmark diagnostic)", parts)
    d12 = {l: data[l]["summary"]["across_seeds"]["d12_converged_count"] for l in lv}
    add("E9", "Fewer than 5 of 10 converge (operative rule); none under the original and exact-action rules",
        [(f"operative < 5 at {LABEL[l]}", d12[l]["2026_11_20_operative"] < 5, f"{d12[l]['2026_11_20_operative']}/10")
         for l in lv]
        + [(f"original and exact-action = 0 at {LABEL[l]}",
            d12[l]["2026_11_12_original_primary_last_iterate"] == 0 == d12[l]["2026_11_19_exact_action_last_iterate"],
            f"{d12[l]['2026_11_12_original_primary_last_iterate']}, {d12[l]['2026_11_19_exact_action_last_iterate']}")
           for l in lv])
    add("E10", "Tail average and last iterate differ by less than 1 bps/yr",
        [(f"|avg − last| < 1 at {LABEL[l]}", abs(g["agent_minus_last_iterate_mean"][l]) < 1,
          f"{g['agent_minus_last_iterate_mean'][l]:.2f}") for l in lv])
    d10 = {l: data[l]["summary"]["d10"] for l in lv}
    add("E11", "D10: $1B SE 0.5 to 0.8, missing a target of about 0.43; met at $10B; n/a at zero and $100M",
        [("$1B SE in [0.5, 0.8]", 0.5 <= d10["aum_1b"]["mean_agent_minus_oracle_se"] <= 0.8,
          f"{d10['aum_1b']['mean_agent_minus_oracle_se']:.2f}"),
         ("$1B misses", not d10["aum_1b"]["met"], f"target {d10['aum_1b']['target_se']:.2f}"),
         ("met at $10B", d10["aum_10b"]["met"], f"{d10['aum_10b']['mean_agent_minus_oracle_se']:.2f} vs {d10['aum_10b']['target_se']:.2f}"),
         ("n/a at zero and $100M", not d10["zero"]["applicable"] and not d10["aum_100m"]["applicable"], "n/a, n/a")])
    sd = data["aum_1b"]["summary"]["across_seeds"]["agent_minus_oracle_mean"]["sd_across_seeds"]
    add("E12", "Across-seed SD of agent − oracle at $1B at least 0.5", [("SD ≥ 0.5", sd >= 0.5, f"{sd:.2f}")])
    return out


def svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    body = buf.getvalue().split("<svg", 1)[1]
    return "<svg" + "\n".join(line.rstrip() for line in body.splitlines())


def fig_split(data):
    lv = config.LEVELS
    x = np.arange(len(lv))
    target = [mean_of(data[l], "split_own_target_minus_always_rebalance_mean") for l in lv]
    timing = [mean_of(data[l], "split_agent_minus_own_target_mean") for l in lv]
    lead = [-data[l]["summary"]["oracle_minus_always_rebalance"]["mean"] for l in lv]
    total = [mean_of(data[l], "agent_minus_oracle_mean") for l in lv]
    fig, ax = plt.subplots(figsize=(10, 3.6), layout="constrained")
    w = 0.18
    ax.bar(x - 1.5 * w, target, w, color="#c0603a", label="Where it heads: own target − always rebalance")
    ax.bar(x - 0.5 * w, timing, w, color="#12667c", label="When it trades: agent − own target")
    ax.bar(x + 0.5 * w, lead, w, color="#8a94a6", label="Minus the oracle's lead over always rebalance")
    for i, l in enumerate(lv):
        seeds = col(data[l]["rows"], "agent_minus_oracle_mean")
        ax.scatter(np.full(len(seeds), i + 1.6 * w), seeds, s=10, color="#25334c", alpha=0.5, zorder=3)
        ax.scatter(i + 1.6 * w, total[i], marker="_", s=250, color="#25334c", zorder=4)
    ax.scatter([], [], s=10, color="#25334c", label="Agent − oracle: seeds (dots) and mean (bar)")
    ax.axhline(0, color="#4a5568", linewidth=0.8)
    ax.set_xticks(x, [LABEL[l] for l in lv])
    ax.set_ylabel("bps/yr")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="y", alpha=0.2)
    return svg(fig)


def fig_vs_references(data):
    lv = config.LEVELS
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), layout="constrained", sharey=False)
    for ax, (key, title) in zip(axes, (("agent_minus_oracle_mean", "Agent − oracle"),
                                       ("agent_minus_always_rebalance_mean", "Agent − always rebalance"),
                                       ("agent_minus_myopic_mean", "Agent − myopic"))):
        for i, l in enumerate(lv):
            v = col(data[l]["rows"], key)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=14, color=COLOR[l], alpha=0.8)
            ax.scatter(i, v.mean(), marker="_", s=300, color="#25334c")
        ax.axhline(0, color="#4a5568", linewidth=0.8)
        ax.set_xticks(range(len(lv)), [LABEL[l] for l in lv])
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("bps/yr, per seed (bar = mean)")
    return svg(fig)


def fig_costs(data):
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.6), layout="constrained", sharex=True)
    for j, l in enumerate(COST_LEVELS):
        for i, (c, key) in enumerate((("c1", "c1_over_true"), ("c2", "c2_over_true"))):
            ax = axes[i, j]
            for r in data[l]["rows"]:
                z = data[l]["logs"][int(r["seed"])]
                ratio = float(r[key])
                final = float(r["final_" + c])
                if ratio and final:
                    true = final / ratio
                    ax.plot(z["checkpoint_block"] / 1e6, z[c] / true, color=COLOR[l], alpha=0.45, linewidth=0.9)
            ax.axhline(1, color="#25334c", linewidth=0.8, linestyle="--")
            ax.axvline(config.HYPERPARAMETERS["cost_rls_after"] / 1e6, color="#a0aec0", linewidth=0.6)
            ax.set_yscale("symlog", linthresh=0.1)
            ax.set_ylim(bottom=0)
            ax.set_title(f"{c} / true, {LABEL[l]}", fontsize=9)
            ax.grid(alpha=0.2)
        axes[1, j].set_xlabel("training block (millions)")
    return svg(fig)


def fig_targets(data):
    best = data["aum_1b"]["summary"]["best_target"]
    fig, ax = plt.subplots(figsize=(4.8, 3.4), layout="constrained")
    for k, l in enumerate(config.LEVELS):
        pts = np.array([[float(x) for x in r["favourite_target"].split()] for r in data[l]["rows"]])
        jitter = (k - 1.5) * 0.003
        ax.scatter(pts[:, 0] + jitter, pts[:, 2] + jitter, s=22, color=COLOR[l], alpha=0.8, label=LABEL[l])
    ax.scatter(best[0], best[2], marker="*", s=220, color="#25334c", label="best target (no cost)", zorder=5)
    ax.set_xlabel(f"{spec.ASSETS[0]} weight")
    ax.set_ylabel(f"{spec.ASSETS[2]} weight ({spec.ASSETS[1]} is the rest)")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.2)
    return svg(fig)


def fig_hold(data):
    lv = config.LEVELS
    fig, ax = plt.subplots(figsize=(4.8, 3.4), layout="constrained")
    for i, l in enumerate(lv):
        v = col(data[l]["rows"], "agent_hold_fraction")
        ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=14, color=COLOR[l])
        ax.scatter(i, data[l]["summary"]["references"]["oracle"]["hold_fraction"], marker="D", s=40,
                   color="#25334c", label="oracle" if i == 0 else None)
    ax.set_xticks(range(len(lv)), [LABEL[l] for l in lv])
    ax.set_ylabel("share of blocks with no trade")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(axis="y", alpha=0.2)
    return svg(fig)


def table(headers, rows, cls=""):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='scroll'><table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def f(x, nd=2, sign=False):
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def build(runs: Path = RUNS, out: Path = OUT) -> Path:
    data = load(runs)
    lv = config.LEVELS
    S = {l: data[l]["summary"] for l in lv}
    A = {l: S[l]["across_seeds"] for l in lv}
    exp = expectations(data)
    n_met = sum(e["verdict"] == "met" for e in exp)
    n_part = sum(e["verdict"] == "partly met" for e in exp)
    n_not = sum(e["verdict"] == "not met" for e in exp)
    prov = S["aum_1b"]["provenance"]
    hp = S["aum_1b"]["hyperparameters"]

    def sd(l, k):
        return A[l][k]["sd_across_seeds"]

    def path_se(l):
        z = np.load(runs / l / "values.npz")
        seeds = [int(r["seed"]) for r in data[l]["rows"]]
        k = (1 - spec.DISCOUNT) / (1 - spec.DISCOUNT ** S[l]["horizon"]) * spec.TRADING_DAYS * spec.BPS
        d = k * (np.mean([z[f"seed{s}_agent"] for s in seeds], axis=0) - z["ref_oracle"])
        return d.std(ddof=1) / np.sqrt(len(d))

    headline = table(
        ("bps/yr, mean of 10 seeds", *[LABEL[l] for l in lv]),
        [("Agent − oracle", *[f"<b>{f(A[l]['agent_minus_oracle_mean']['mean'], sign=True)}</b>" for l in lv]),
         ("&nbsp;&nbsp;across-seed SD", *[f(sd(l, 'agent_minus_oracle_mean')) for l in lv]),
         ("&nbsp;&nbsp;path SE of the seed-averaged gap", *[f(path_se(l)) for l in lv]),
         ("&nbsp;&nbsp;seeds below the oracle",
          *[f"{int((col(data[l]['rows'], 'agent_minus_oracle_mean') < 0).sum())}/10" for l in lv]),
         ("Agent − always rebalance", *[f(A[l]['agent_minus_always_rebalance_mean']['mean'], sign=True) for l in lv]),
         ("Agent − myopic oracle", *[f(A[l]['agent_minus_myopic_mean']['mean'], sign=True) for l in lv]),
         ("Agent − trade once, then hold", *[f(A[l]['agent_minus_trade_once_then_hold_mean']['mean'], sign=True) for l in lv]),
         ("Split: when it trades (agent − own target)",
          *[f(A[l]['split_agent_minus_own_target_mean']['mean'], sign=True) for l in lv]),
         ("Split: where it heads (own target − always rebalance)",
          *[f(A[l]['split_own_target_minus_always_rebalance_mean']['mean'], sign=True) for l in lv]),
         ("Split: minus the oracle's lead over always rebalance",
          *[f(-S[l]['oracle_minus_always_rebalance']['mean'], sign=True) for l in lv]),
         ("Favourite target's utility loss (benchmark diagnostic)",
          *[f(A[l]['favourite_target_utility_loss_bps']['mean']) for l in lv]),
         ("Tail average − last iterate", *[f(A[l]['agent_minus_last_iterate_mean']['mean'], sign=True) for l in lv])])
    refs = table(
        ("Policy", *[LABEL[l] for l in lv]),
        [(name, *[f(S[l]["references"][key]["mean_value_bps"], 1) for l in lv])
         for name, key in (("Oracle", "oracle"), ("Myopic oracle (ρ = 0 when choosing)", "myopic"),
                           ("Always rebalance to the best target", "always_rebalance"),
                           ("Trade once, then hold", "trade_once_then_hold"))]
        + [("Agent (tail-averaged θ), mean of seeds",
            *[f(np.mean([p["agent"]["mean_value_bps"] for p in S[l]["per_seed"]]), 1) for l in lv]),
           ("Oracle − always rebalance (paired SE)",
            *[f"{f(S[l]['oracle_minus_always_rebalance']['mean'])} ({f(S[l]['oracle_minus_always_rebalance']['se'])})" for l in lv]),
           ("Oracle − myopic (paired SE)",
            *[f"{f(S[l]['oracle_minus_myopic']['mean'])} ({f(S[l]['oracle_minus_myopic']['se'])})" for l in lv])])
    behaviour = table(
        ("", *[LABEL[l] for l in lv]),
        [("Agent hold fraction, mean (seed range)",
          *[f"{f(A[l]['agent_hold_fraction']['mean'])} ({f(A[l]['agent_hold_fraction']['min'])} to {f(A[l]['agent_hold_fraction']['max'])})" for l in lv]),
         ("Oracle hold fraction", *[f(S[l]["references"]["oracle"]["hold_fraction"]) for l in lv]),
         ("Agent one-way turnover per block", *[f(A[l]["agent_turnover_per_block"]["mean"], 4) for l in lv]),
         ("Oracle one-way turnover per block", *[f(S[l]["references"]["oracle"]["turnover_per_block"], 4) for l in lv]),
         ("Agent cost, bps per block", *[f(A[l]["agent_cost_bps_per_block"]["mean"], 3) for l in lv]),
         ("Oracle cost, bps per block", *[f(S[l]["references"]["oracle"]["cost_bps_per_block"], 3) for l in lv]),
         ("c2 / true, median (seed range)", "n/a",
          *[f"{f(np.nanmedian(col(data[l]['rows'], 'c2_over_true')))} ({f(np.nanmin(col(data[l]['rows'], 'c2_over_true')))} to {f(np.nanmax(col(data[l]['rows'], 'c2_over_true')))})" for l in COST_LEVELS]),
         ("c1 / true, median (seed range)", "n/a",
          *[f"{f(np.nanmedian(col(data[l]['rows'], 'c1_over_true')))} ({f(np.nanmin(col(data[l]['rows'], 'c1_over_true')))} to {f(np.nanmax(col(data[l]['rows'], 'c1_over_true')))})" for l in COST_LEVELS])])
    conv_keys = (("2026_11_12_original_primary_last_iterate", "Original (2026-11-12), last iterate"),
                 ("2026_11_19_exact_action_last_iterate", "Exact action (2026-11-19), last iterate"),
                 ("2026_11_19_tail_average", "Exact action, tail average"),
                 ("2026_11_20_operative", "<b>Operative (2026-11-20), tail average</b>"))
    conv = table(("D12 version", *[LABEL[l] for l in lv]),
                 [(name, *[f"{A[l]['d12_converged_count'][k]}/10" for l in lv]) for k, name in conv_keys]
                 + [("D10: mean paired SE vs target",
                     *[f"{f(S[l]['d10']['mean_agent_minus_oracle_se'])} vs {f(S[l]['d10']['target_se'])}, "
                       + ("met" if S[l]["d10"]["met"] else ("missed" if S[l]["d10"]["applicable"] else "n/a")) for l in lv])])
    badge = {"met": "ok", "partly met": "part", "not met": "no"}
    exp_rows = [(e["id"], html.escape(e["expectation"]), f"<span class='{badge[e['verdict']]}'>{e['verdict']}</span>",
                 "<br>".join(("✓ " if p["met"] else "✗ ") + html.escape(f"{p['clause']}: {p['observed']}") for p in e["parts"]))
                for e in exp]
    seed_tables = []
    for l in lv:
        rows = [(r["seed"], f(float(r["agent_minus_oracle_mean"]), sign=True) + f" ± {f(float(r['agent_minus_oracle_se']))}",
                 f(float(r["agent_minus_always_rebalance_mean"]), sign=True),
                 f(float(r["split_agent_minus_own_target_mean"]), sign=True),
                 f(float(r["split_own_target_minus_always_rebalance_mean"]), sign=True),
                 r["favourite_target"], f(float(r["agent_hold_fraction"])),
                 "n/a" if r["c1_over_true"] in ("", "None") else f(float(r["c1_over_true"])),
                 "n/a" if r["c2_over_true"] in ("", "None") else f(float(r["c2_over_true"])),
                 "yes" if r["2026_11_20_operative"] == "True" else "no",
                 r["converged_from_block"] or "") for r in data[l]["rows"]]
        seed_tables.append(f"<details><summary>{LABEL[l]}: per seed</summary>" + table(
            ("Seed", "Agent − oracle ± SE", "− always rebalance", "timing", "target", "Favourite target",
             "Hold", "c1/true", "c2/true", "Converged", "From block"), rows) + "</details>")

    payload = {"levels": {l: {"across_seeds": A[l], "references": S[l]["references"], "d10": S[l]["d10"],
                              "oracle_minus_always_rebalance": S[l]["oracle_minus_always_rebalance"],
                              "seed_averaged_path_se": path_se(l)} for l in lv},
               "expectations": exp, "provenance": prov, "hyperparameters": hp}
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8", newline="\n")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TC with drift · final runs</title>
<style>
:root{{--ink:#182b3f;--muted:#5a6b7d;--bg:#f5f7fa;--card:#fff;--line:#dbe4ec;--head:#edf5fa;--accent:#12667c}}
body{{font:15px/1.55 system-ui,sans-serif;color:var(--ink);background:var(--bg);margin:0}}
main{{max-width:1120px;margin:auto;padding:1.5rem 1rem 4rem}}
header,section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1.3rem;margin-bottom:1rem}}
h1,h2{{line-height:1.2;margin:.2rem 0 .7rem}} h2{{font-size:1.2rem}} .eyebrow{{color:var(--accent);font-size:.8rem;font-weight:700;letter-spacing:.03em}}
.scroll{{overflow-x:auto}} table{{width:100%;border-collapse:collapse;font-size:.86rem}}
td,th{{padding:.45rem .55rem;text-align:right;border-bottom:1px solid #e6edf2;vertical-align:top}} th{{background:var(--head)}}
td:first-child,th:first-child{{text-align:left}} table.exp td{{text-align:left}} table.exp td:nth-child(4){{font-size:.8rem;color:var(--muted)}}
.ok,.part,.no{{padding:.1rem .45rem;border-radius:4px;font-weight:600;white-space:nowrap}}
.ok{{background:#e3f3ea;color:#1f6b3e}} .part{{background:#fdf1dc;color:#8a5a12}} .no{{background:#fbe3e1;color:#9b2c24}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.7rem;margin:.8rem 0}}
.kpi{{border:1px solid var(--line);border-radius:8px;padding:.7rem}} .kpi b{{font-size:1.5rem;display:block}} .kpi span{{color:var(--muted);font-size:.82rem}}
svg{{width:100%;height:auto}} .pair{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem;align-items:start}} .pair svg{{max-width:560px}} code{{background:var(--head);padding:.1rem .3rem;border-radius:3px}} details{{margin:.5rem 0}} summary{{cursor:pointer;font-weight:600}}
.note{{color:var(--muted);font-size:.88rem}}
</style></head><body><main>
<header><div class="eyebrow">MODEL-FREE Q-LEARNING · TRANSACTION COSTS WITH DRIFT · POST HOC · FULL SCALE</div>
<h1>Final agent runs: 4 cost levels × 10 seeds</h1>
<p><b>In plain terms:</b> the agent loses to the oracle at every cost level, mostly because it heads for a slightly wrong portfolio.
At zero cost, $100M and $1B it is {f(-A['aum_1b']['agent_minus_oracle_mean']['mean'], 0)} to {f(-A['zero']['agent_minus_oracle_mean']['mean'], 0)} bps/yr behind the oracle and also loses to always rebalancing.
At $10B, where trading costs matter most, it learns when to trade: {f(A['aum_10b']['agent_minus_always_rebalance_mean']['mean'], 1, True)} bps/yr against always rebalancing and
{f(-A['aum_10b']['agent_minus_oracle_mean']['mean'], 1)} behind the oracle. Of the 12 expectations registered before the runs, {n_met} are met, {n_part} partly met and {n_not} not met.</p>
<div class="kpis">
<div class="kpi"><b>{f(A['aum_1b']['agent_minus_oracle_mean']['mean'], 1, True)}</b><span>agent − oracle at $1B (headline), bps/yr; SD across seeds {f(sd('aum_1b','agent_minus_oracle_mean'), 1)}</span></div>
<div class="kpi"><b>{f(A['aum_10b']['split_agent_minus_own_target_mean']['mean'], 1, True)}</b><span>value of the agent's trade timing at $10B (agent − rebalance to its own target)</span></div>
<div class="kpi"><b>{f(A['aum_1b']['split_own_target_minus_always_rebalance_mean']['mean'], 1, True)}</b><span>cost of where it heads at $1B (own target − best target)</span></div>
<div class="kpi"><b>{n_met} / {n_part} / {n_not}</b><span>expectations met / partly met / not met</span></div>
</div>
<p class="note">Values are expected discounted mean-variance utility over H = {S['aum_1b']['horizon']} quarterly blocks, in bps/yr, on m = {S['aum_1b']['m']:,} paired paths and starts shared by every policy.
γ = {S['aum_1b']['gamma']:g}, ρ = {spec.DISCOUNT} per block, three ETFs ({", ".join(spec.ASSETS)}), 63-day blocks, 2% targets. "Agent" is greedy on θ averaged over the last 10% of training.
Built by <code>python -m src.tc_drift.report</code> from committed files only. Design: <code>docs/methods/TC_DRIFT_DESIGN.md</code>; results: <code>results/tc-drift/runs/</code>.</p></header>

<section><h2>Where the gap to the oracle comes from</h2>
<p>The agent's gap to the oracle splits exactly into three paired terms (design §7, TC-3), checked on every path: where the agent heads (rebalancing to its own favourite target against rebalancing to the best target), when it trades (the agent against rebalancing to its own target), and the oracle's own lead over always rebalancing.
At the three lower levels almost the whole gap is where it heads. At $10B the timing term is large and positive.</p>
{fig_split(data)}
{headline}
<p class="note">Across-seed SD is the spread of the 10 seeds' gaps; the path SE is the paired SE of the seed-averaged policy value against the oracle on the shared paths. The favourite target's utility loss uses μ and Σ and is a benchmark diagnostic, not available to the agent.</p></section>

<section><h2>Every seed against the references</h2>{fig_vs_references(data)}{refs}</section>

<section><h2>Registered expectations (E1 to E12)</h2>
<p>Registered by Albert on 2026-11-20 in design §7 before any final run; William added none. Each clause is checked as written. Under "what would change the story": E1 failed away from zero, not toward it; at $10B the timing term is positive while the target error is small, so there the agent learned both when to trade and roughly where to head.</p>
{table(("#", "Expectation", "Verdict", "Clauses (observed)"), exp_rows, "exp")}</section>

<section><h2>Where the agent heads, and how often it trades</h2>
<p>Left: each seed's favourite target (highest learned value θ·φ over the 1,326 targets); the star is the best no-cost target, {best_fmt(S)}. Right: share of blocks in which each seed's policy does not trade, against the oracle's (diamonds), on the evaluation paths.</p>
<div class="pair"><div>{fig_targets(data)}</div><div>{fig_hold(data)}</div></div>{behaviour}</section>

<section><h2>Learned costs</h2>
<p>Learned cost coefficients over training, as a ratio to the truth (one line per seed; dashed line = truth; grey line = switch from constant-step burn-in to recursive least squares at block {hp['cost_rls_after']:,}). c2 converges near the truth at $1B and $10B; c1 is often clamped at zero (D11) or far too high.</p>
{fig_costs(data)}</section>

<section><h2>Convergence (D12) and evaluation precision (D10)</h2>
<p>D12 was restated twice after the tuning pilot and frozen on 2026-11-20, before these runs; all four versions are reported. Part IV counts (block, episode, daily observations at which the operative rule first holds through the end) are in <code>per_seed.csv</code>.</p>
{conv}
<p class="note">D10 asks for the agent − oracle paired SE to be below a third of the oracle's lead over always rebalancing. It does not apply where that lead is not distinguishable from zero.</p>
{"".join(seed_tables)}</section>

<section><h2>Settings and provenance</h2>
<p>{len(data['aum_1b']['rows'])} seeds per level, {S['aum_1b']['n_blocks']:,} training blocks per run (restart every {config.RESTART_EVERY} blocks), master seed {prov['seed_plan']['master_seed']}.
Hyperparameters (tuned on seed {spec.TUNING_MASTER_SEED} at $1B, design §9): α_value = {hp['alpha_value']:g}, α_cost = {hp['alpha_cost']:g}, β = {hp['beta']:.3e} (ε reaches 1% at halfway), RLS on the cost block after {hp['cost_rls_after']:,} updates with prior {hp['cost_rls_prior']:g}·I, target refresh every {hp['target_refresh_every']:,} blocks, l2 = {hp['l2']:g}.</p>
<p>Commit <code>{prov['commit'][:10]}</code>, clean tree, Python {prov['python']}, numpy {prov['numpy']}, {prov['machine']}, BLAS pinned to one thread; about {f(np.mean([float(r['train_seconds']) for r in data['aum_1b']['rows']]) / 60, 0)} minutes a run with seven in parallel.
Per-path values were recomputed independently from <code>values.npz</code> and match the summaries within 1e-13 (<code>results/tc-drift/runs/README.md</code>).</p>
<p class="note">Post hoc relative to <code>baseline-v1</code>. Benchmark information (the true costs, the target error from μ and Σ) informed the learning-rule choice, the tail averaging and the D12 restatements (design §14 risk 5); the agent itself never sees μ, Σ, κ, λ or the oracle.</p></section>
</main></body></html>
"""
    (out / "report.html").write_text(page, encoding="utf-8", newline="\n")
    return out / "report.html"


def best_fmt(S):
    b = S["aum_1b"]["best_target"]
    return ", ".join(f"{a} {w:.2f}" for a, w in zip(spec.ASSETS, b))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=Path, default=RUNS)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    print(f"Wrote {build(a.runs, a.out)}")
