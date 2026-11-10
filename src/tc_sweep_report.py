"""Build the standalone transaction-cost sweep report from completed runs."""

import argparse
import hashlib
import html
import io
import json
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib.pyplot as plt
import numpy as np

from src.action_space.grid import generate_weight_grid
from src.environment import ReturnSimulator, SeedPlan, load_calibration
from src.evaluate import objective
from src.train import grid_fingerprint
from src.transaction_costs import evaluate_policy, switching_cost


def build(results, out, allow_reduced=False):
    names = ("no-cost", "low-cost", "anchor", "high-cost")
    runs = [json.loads((results / f"{name}.json").read_text()) for name in names]
    first = runs[0]
    shared = (
        "git_commit", "grid_sha256", "calibration_sha256", "seed_plan", "gamma",
        "block_length", "steps_per_seed", "evaluation_paths", "evaluation_blocks",
        "evaluation_discount", "evaluation_start_indices",
    )
    if any(run[key] != first[key] for run in runs for key in shared):
        raise ValueError("Cost workers used different shared settings")
    n = first["seed_plan"]["n_runs"]
    full_scale = n == 10 and first["steps_per_seed"] == 13_000_000
    if not full_scale and not allow_reduced:
        raise ValueError("Expected the full registered budget")
    baseline_path = Path("results/baseline_g5_grid20/evaluation/summary.json")
    baseline = json.loads(baseline_path.read_text())
    baseline_seeds = baseline["per_seed"][:n]
    mu, sigma = load_calibration()
    calibration = hashlib.sha256(np.concatenate([mu, sigma.ravel()]).tobytes()).hexdigest()
    if calibration != first["calibration_sha256"]:
        raise ValueError("Evaluation calibration does not match training")
    grid = generate_weight_grid(len(mu), 0.2)
    if grid_fingerprint(grid) != first["grid_sha256"]:
        raise ValueError("Evaluation grid does not match training")
    gamma = first["gamma"]
    horizon, blocks = first["evaluation_blocks"], first["block_length"]
    discount = first["evaluation_discount"]
    rng = SeedPlan(first["seed_plan"]["master_seed"], n).evaluation_rng()
    starts = rng.integers(len(grid), size=first["evaluation_paths"])
    np.testing.assert_array_equal(starts, first["evaluation_start_indices"])
    paths = ReturnSimulator(mu, sigma, rng).sample_episodes(
        horizon * len(starts), blocks
    ).reshape(horizon, len(starts), blocks, len(mu))
    utility = objective(grid, mu, sigma, gamma)
    reward_key = "expected_net_ce_bps_per_year"
    realized_key = "realized_net_ce_bps_per_year"
    rows = []
    for name, run, expected_level in zip(names, runs, (0, 1, 2, 5), strict=True):
        variant = run["variants"][name]
        seeds = variant["seeds"]
        if [seed["seed"] for seed in seeds] != list(range(n)):
            raise ValueError(f"{name} has incomplete or mismatched seeds")
        if variant["settings"] != {"reference_cost_bps": expected_level, "discount": 0.95, "alpha": None}:
            raise ValueError(f"{name} differs from the registered cost sweep")
        cost = switching_cost(grid, expected_level) / blocks
        old = [
            evaluate_policy(np.full(len(grid), seed["learned_action"]), starts, paths,
                            grid, cost, utility, gamma, discount)
            for seed in baseline_seeds
        ]
        paired = [new[reward_key] - ref[reward_key] for new, ref in zip(seeds, old, strict=True)]
        rows.append({
            "cost": expected_level,
            "benchmarks": variant["benchmarks"],
            "seeds": [
                {"seed": seed["seed"], "learned": {k: v for k, v in seed.items() if k != "policy"},
                 "baseline": ref, "paired_net_ce_delta_bps_per_year": delta}
                for seed, ref, delta in zip(seeds, old, paired, strict=True)
            ],
            "median_learned_net_ce": median(s[reward_key] for s in seeds),
            "median_learned_realized_ce": median(s[realized_key] for s in seeds),
            "median_baseline_net_ce": median(s[reward_key] for s in old),
            "median_baseline_realized_ce": median(s[realized_key] for s in old),
            "paired_delta_mean": mean(paired),
            "paired_delta_se": stdev(paired) / n**0.5 if n > 1 else None,
            "positive_pairs": sum(d > 1e-8 for d in paired),
            "median_oracle_gap": median(s["oracle_gap_bps_per_year"] for s in seeds),
            "median_learned_turnover": median(s["one_way_turnover_per_path"] for s in seeds),
            "median_baseline_turnover": median(s["one_way_turnover_per_path"] for s in old),
            "median_trades": median(s["trades_per_path"] for s in seeds),
            "median_coverage": median(s["visited_pairs"] / s["possible_pairs"] for s in seeds),
            "stable_seeds": sum(s["checkpoint_stable_through_end"] for s in seeds),
        })

    out.mkdir(parents=True, exist_ok=True)
    original = median(s["ce_loss_vs_best_discrete_bps"] for s in baseline_seeds)
    result = {"full_scale": full_scale, "settings": {k: first[k] for k in shared if k != "evaluation_start_indices"},
              "baseline_original_median_learning_gap_bps_per_year": original, "costs": rows}
    (out / "data.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    costs = [r["cost"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
    ax, turn = axes
    for label, key, color in (
        ("Cost-aware Q-learning", "median_learned_net_ce", "#12667c"),
        ("Baseline-v1, charged the same costs", "median_baseline_net_ce", "#7c5d90"),
    ):
        ax.plot(costs, [r[key] for r in rows], marker="o", label=label, color=color)
    ax.plot(costs, [r["benchmarks"]["grid_oracle"][reward_key] for r in rows],
            marker="o", label="Dynamic grid oracle", color="#25334c", linestyle="--")
    ax.set_ylabel("Expected net CE (bp/year)")
    ax.legend(frameon=False, fontsize=8)
    for r in rows:
        turn.scatter([r["cost"]] * n, [s["paired_net_ce_delta_bps_per_year"] for s in r["seeds"]],
                     color="#12667c", alpha=0.45, s=24)
        turn.scatter(r["cost"], r["paired_delta_mean"], marker="_", s=160, color="#1c2a3d")
    turn.axhline(0, color="#718096", linewidth=0.8)
    turn.set_ylabel("Paired gain vs. baseline-v1 (bp/year)")
    for axis in axes:
        axis.set_xticks(costs)
        axis.set_xlabel("Cost of a 20-point transfer (bp)")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    image = io.StringIO()
    fig.savefig(image, format="svg")
    plt.close(fig)
    svg_body = image.getvalue().split("<svg", 1)[1]
    svg = "\n".join(line.rstrip() for line in svg_body.splitlines())

    def table(headers, values):
        head = "".join(f"<th scope='col'>{html.escape(h)}</th>" for h in headers)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in line) + "</tr>"
                       for line in values)
        return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"

    headline = table(
        ("Cost (bp)", "Learned net CE", "Baseline net CE", "Paired gain ± SE", "Better seeds",
         "Grid oracle", "Myopic", "Median oracle gap"),
        [(r["cost"], f"{r['median_learned_net_ce']:.1f}", f"{r['median_baseline_net_ce']:.1f}",
          f"{r['paired_delta_mean']:+.1f} ± {r['paired_delta_se']:.1f}" if r["paired_delta_se"] is not None
          else f"{r['paired_delta_mean']:+.1f}", f"{r['positive_pairs']}/{n}",
          f"{r['benchmarks']['grid_oracle'][reward_key]:.1f}", f"{r['benchmarks']['myopic'][reward_key]:.1f}",
          f"{r['median_oracle_gap']:.1f}") for r in rows],
    )
    behavior = table(
        ("Cost (bp)", "Learned turnover", "Baseline turnover", "Learned trades",
         "Pair coverage", "Policy stable", "Realized CE, learned", "Realized CE, baseline"),
        [(r["cost"], f"{r['median_learned_turnover']:.2f}", f"{r['median_baseline_turnover']:.2f}",
          f"{r['median_trades']:.1f}", f"{r['median_coverage']:.1%}", f"{r['stable_seeds']}/{n}",
          f"{r['median_learned_realized_ce']:.1f}", f"{r['median_baseline_realized_ce']:.1f}")
         for r in rows],
    )
    anchor = rows[2]
    scale = "Full-scale" if full_scale else "Reduced pipeline check"
    report = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quadratic transaction-cost sweep · Q-learning</title>
<style>
body{{font:16px/1.55 system-ui,sans-serif;color:#182b3f;background:#f5f7fa;margin:0}}
main{{max-width:1120px;margin:auto;padding:2rem 1.25rem 4rem}}
header,section{{background:white;border:1px solid #dbe4ec;border-radius:10px;padding:1.4rem;margin-bottom:1rem}}
h1,h2{{line-height:1.2;margin:.2rem 0 .75rem}} .eyebrow{{color:#12667c;font-size:.85rem;font-weight:700}}
.scroll{{overflow-x:auto}} table{{width:100%;border-collapse:collapse;font-size:.9rem}}
td,th{{padding:.6rem;text-align:right;border-bottom:1px solid #e1e8ee}} th{{background:#edf5fa}}
svg{{width:100%;height:auto}} a{{color:#12667c}} code{{background:#edf5fa;padding:.15rem .3rem}}
</style><main>
<header><div class="eyebrow">MODEL-FREE Q-LEARNING · POST HOC · {scale}</div>
<h1>Quadratic transaction-cost sweep</h1>
<p>{n} seeds × {first['steps_per_seed']:,} training blocks per cost level. Twenty-day blocks, a 20% portfolio grid,
γ = {gamma:g}, discount {discount:g}, and a 1/N learning rate per state–action pair.</p>
<p>At the 2 bp anchor, the cost-aware policy changed expected net certainty equivalent by
<strong>{anchor['paired_delta_mean']:+.1f} bp/year</strong> on average versus the original baseline policies
charged the same costs. It improved on {anchor['positive_pairs']} of {n} paired seeds and finished
{anchor['median_oracle_gap']:.1f} bp/year below the grid oracle at the median.</p>
<p>Training commit <code>{html.escape(first['git_commit'] or 'unknown')}</code>.
<a href="data.json">Download the comparison data</a>.</p></header>
<section><h2>Performance against baseline-v1</h2>
<p>All CE values are in basis points per year. Learned and baseline columns are medians across seeds.
Each pair reuses the same starting allocations and fresh return paths. Positive gain favors the cost-aware agent.</p>
{headline}<svg{svg}
<p>In its original stateless evaluation, baseline-v1 at γ = 5 had a median learning gap of
{original:.1f} bp/year to the best grid portfolio and a {baseline['discretization_gap']['ce_loss_bps']:.1f} bp/year
discretization gap. The comparison above replays each of those learned policies as a fixed target,
charging the initial move at the same cost as the new agent. It compares the policies under the
transaction-cost objective rather than subtracting scores from different objectives.</p></section>
<section><h2>Trading and training behavior</h2>
<p>Turnover is cumulative one-way turnover per 100-block path. Trades and turnover are medians across seeds.
Policy stability means the full greedy policy stayed unchanged at checkpoints through the final 10% of training.</p>
{behavior}
<p>The zero-cost control isolates a key limitation of this run. At zero cost, the optimal target is the
same from every starting allocation. The stateful table still learns a separate action value from each
starting allocation. Here, the median seed visited {rows[0]['median_coverage']:.1%} of possible pairs, no
seed met the checkpoint stability criterion, and the median oracle gap was
{rows[0]['median_oracle_gap']:.1f} bp/year. The learned policy was already far behind before trading
charges entered the reward.</p></section>
<section><h2>How to read the comparison</h2>
<p>The grid oracle and the myopic benchmark use the same grid and costs. The oracle values future blocks;
the myopic benchmark accounts only for the next block when choosing an action. Both are evaluated
with discount 0.95. Realized scores use {len(starts):,} fresh return paths.</p>
<p>This comparison changes state representation and discount relative to baseline-v1, in addition to
training with costs. Differences between the four cost levels isolate the configured cost.
The model treats the previous target as current holdings and abstracts from drift.
With stationary returns, trading behavior may be concentrated near the starting allocation.</p>
</section></main></html>"""
    (out / "report.html").write_text(report, encoding="utf-8")
    print(f"Wrote {out / 'report.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/tc-quadratic-cost"))
    parser.add_argument("--out", type=Path, default=Path("reports/tc-quadratic-cost"))
    parser.add_argument("--allow-reduced", action="store_true")
    args = parser.parse_args()
    build(args.results, args.out, args.allow_reduced)
