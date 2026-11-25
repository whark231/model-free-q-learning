"""Build the self-contained slide deck docs/report/slides/slides.html.

Run from anywhere:  python docs/report/slides/build_slides.py
Figures are read from docs/report/figures/*.png (make_figures.py) and embedded
as base64, so slides.html has no external files.
"""
from __future__ import annotations

import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"


def img(name: str) -> str:
    data = base64.b64encode((FIG / f"{name}.png").read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{name}">'


def stat(number: str, label: str, tone: str = "") -> str:
    return f'<div class="stat {tone}"><div class="num">{number}</div><div class="lab">{label}</div></div>'


def slide(title: str, body: str, tag: str = "", kicker: str = "") -> str:
    tag_html = f'<span class="tag">{tag}</span>' if tag else ""
    kick = f'<div class="kicker">{kicker}</div>' if kicker else ""
    return f'<section class="slide">{tag_html}{kick}<h2>{title}</h2>{body}</section>'


def wide(fig: str, stats: list[str], note: str = "") -> str:
    note_html = f'<p class="note">{note}</p>' if note else ""
    return f'<div class="wide"><div class="figure">{img(fig)}</div><div class="stats">{"".join(stats)}</div>{note_html}</div>'


def split(fig: str, left: str, ratio: str = "") -> str:
    return f'<div class="split {ratio}"><div class="text">{left}</div><div class="figure">{img(fig)}</div></div>'


POST = "post hoc"
slides = []

# 1 ------------------------------------------------------------------ title
slides.append(f'''<section class="slide title">
<div class="kicker">Data Science for Finance · Fall 2026 · Albert Aldrich and William Harkless</div>
<h1>Can an agent that only sees rewards find a near-optimal portfolio?</h1>
<div class="split wide-fig"><div class="text">
<p class="lead">Nine ETFs, a simulator calibrated once, a Q-learning agent choosing among 1,287 long-only portfolios.</p>
<div class="stats col">
{stat("19.8", "bps/yr behind the optimum, median across seeds, out of 530.5 available", "teal")}
{stat("310.9", "bps/yr behind for an equal-weight portfolio", "grey")}
</div></div>
<div class="figure">{img("fig_gap_ladder")}</div></div>
</section>''')

# 2 ------------------------------------------------------------------- data
slides.append(slide(
    "The environment: 20 years of daily data, estimated once and frozen",
    split("fig_correlation", f'''
<ul>
<li>SPY, EFA, EEM · IEF, LQD, TIP · GLD, VNQ, DBC</li>
<li>Adjusted Close, simple excess returns, one daily frequency</li>
<li>7 Feb 2006 to 31 Jul 2026, inner-joined, nothing imputed</li>
<li>Gaussian simulator with μ and Σ fixed; the agent sees simulated rewards only</li>
</ul>
<div class="stats">
{stat("5,152", "daily observations")}
{stat("0.07", "SPY–GLD correlation; Sharpe 0.55 and 0.50")}
{stat("−0.29", "SPY–IEF correlation")}
</div>'''),
    kicker="Part I · data and calibration"))

# 3 ------------------------------------------------------------- benchmarks
slides.append(slide(
    "Two benchmarks under the agent's own constraints",
    wide("fig_benchmarks_by_gamma", [
        stat("530.5", "bps/yr: value of the long-only optimum at γ = 5"),
        stat("5.7", "bps/yr discretization gap on the 20% grid (11.7 at 25%, 1.3 at 10%)"),
        stat("3 of 9", "ETFs held by the optimum: SPY, IEF, GLD"),
    ], "U(w) = w′μ − (γ/2)·w′Σw, long-only and fully invested for theory and agent alike. "
       "The grid is built from the asset count and the step only; the optimum is never inserted."),
    kicker="Part II · theory"))

# 4 ------------------------------------------------------------------ agent
slides.append(slide(
    "The agent: one value per portfolio, updated with averaged rewards",
    split("fig_epsilon", f'''
<ul>
<li><b>Stateless</b>, 1,287 actions (20% grid)</li>
<li><b>Reward</b>: 20-day realized mean − (γ/2) × realized variance, so E[reward] = U(w)</li>
<li><b>Update</b>: Q(a) ← Q(a) + (1/N(a))·[R − Q(a)], discount 0</li>
<li><b>ε-greedy</b>, ε = e<sup>−βt</sup>, 1% at the halfway episode</li>
<li>Settings and convergence rule fixed before any full-scale run</li>
</ul>
<div class="stats">
{stat("13M", "episodes per seed (260M simulated days)")}
{stat("10", "seeds")}
{stat("≈1,097", "random tries per portfolio")}
</div>'''),
    kicker="Part III · design"))

# 5 ---------------------------------------------------------------- learned
slides.append(slide(
    "What it learned: SPY and GLD on every seed, usually a bond fund",
    wide("fig_weights", [
        stat("14.1", "bps/yr median learning gap to the best grid portfolio (range 0 to 47.3)", "teal"),
        stat("3 of 10", "seeds pick the best grid portfolio exactly"),
        stat("0.331", "median L1 distance to the continuous optimum"),
    ]),
    kicker="Pre-registered result · learned portfolios, γ = 5"))

# 6 ------------------------------------------------------------ convergence
slides.append(slide(
    "Convergence: 9 of 10 seeds meet the rule set in advance",
    wide("fig_convergence_times", [
        stat("9 of 10", "meet the primary rule: greedy portfolio unchanged through the end (median 185.1M days)"),
        stat("7 of 10", "stay within 25 bps/yr of the best grid portfolio for good, by 25.8M days", "teal"),
        stat("47.1M", "days: median time to settle on the final portfolio (post hoc measure)"),
    ], "Statistical and economic convergence differ: one seed meets the primary rule while 47.3 bps/yr from the best grid portfolio."),
    kicker="Pre-registered result · convergence, defined before the results"))

# 7 ------------------------------------------------------------------ noise
slides.append(slide(
    "Why not exact: the top portfolios are 10 bps/yr apart and the noise is 3,900",
    wide("fig_noise", [
        stat("10.5", "bps/yr between the best and second-best grid portfolio", "orange"),
        stat("≈3,900", "bps/yr standard deviation of one episode's reward"),
        stat("1.1M", "tries of each needed to separate them; the best got 1,118 to 4,282"),
    ], "The table ranks the grid well overall (Spearman 0.821) and poorly at the very top (35% of the true top ten found). (Post hoc diagnostic.)"),
    kicker="What prevented exact recovery"))

# 8 ------------------------------------------------------------- resolution
slides.append(slide(
    "A finer grid trades discretization error for learning error",
    wide("fig_convergence_resolution", [
        stat("14.3 vs 19.8", "median total gap, 25% grid vs 20% grid (pre-registered)"),
        stat("−0.6", "bps/yr paired change from 20% to 10% grid at γ = 5 (SE 3.7); +18.0 and +4.6 at γ = 3 and 8", "orange"),
        stat("0 of 30", "seeds converge on the 10% grid"),
    ]),
    tag="10% grid: post hoc", kicker="Part VI · action-space design"))

# 9 ------------------------------------------------------------- robustness
slides.append(slide(
    "Longer episodes and more data cut the gap; slower exploration mostly does not",
    wide("fig_robustness", [
        stat("17.6 → 5.1", "mean gap at γ = 5 with 63-day episodes"),
        stat("17.6 → 8.3", "with twice the episodes"),
        stat("5, 8, 9 of 10", "seeds exactly right at γ = 5, 3, 8 with all three changes"),
    ], "Pre-registered, unlike the runs in the chart: risk aversion γ = 3 and 8, with median gaps 10.0 and 5.7 and learned portfolios that follow the optimum toward bonds."),
    tag=POST, kicker="Part VI · economic and statistical assumptions"))

# 10 ----------------------------------------------------------------- FA
slides.append(slide(
    "Spline function approximation: better ranking, worse top pick on the large grid",
    wide("fig_fa", [
        stat("43.5 → 15.7", "20% grid, γ = 5: constant-α table vs splines (baseline 17.6)", "teal"),
        stat("11.7 → 78.8", "10% grid, γ = 5: 1/N table vs constant-α splines (paired +67.1, SE 9.1)", "orange"),
        stat("0.976 vs 0.543", "Spearman on the 10% grid at γ = 5, splines vs table; at the median splines find none of the true top ten"),
    ], "No constant-α table was run on the 10% grid, so that comparison also changes the step rule; on the 20% grid the constant "
       "step alone costs the table 25.9 bps/yr at γ = 5. Step sizes were tuned by scoring final portfolios on simulated draws."),
    tag=POST, kicker="Bonus extension 1 · a negative result on the 10% grid"))

# 11 ------------------------------------------------------------ TC design
slides.append(slide(
    "Transaction costs: a redesign after a first design that learned nothing",
    wide("fig_tc_schematic", [
        stat("56.6 vs 524.8", "first design at zero cost, learned vs grid oracle: 1.66M (holding, target) pairs, 57.1% ever tried", "orange"),
        stat("3 ETFs, 1,326 targets", "redesign: SPY, IEF, GLD; holdings drift over 63-day blocks; costs sized for $100M, $1B, $10B"),
        stat("1.78 and 7.79", "bps/yr: the oracle's lead over always rebalancing at $1B and $10B"),
    ], "A numerical oracle is the benchmark. The agent never sees μ, Σ or the cost parameters."),
    tag=POST, kicker="Bonus extension 2 · design"))

# 12 ----------------------------------------------------------- TC results
slides.append(slide(
    "The agent's gap to the oracle narrows as costs rise, from 13.1 bps/yr to 2.8",
    wide("fig_tc_split", [
        stat("−13.1 · −13.1 · −10.0 · −2.8", "agent − oracle, bps/yr, at zero, $100M, $1B, $10B; negative on all 40 seeds", "orange"),
        stat("+6.2", "bps/yr from trade timing at $10B, against the oracle's 7.8 lead over always rebalancing; the agent beats always rebalancing by 5.0", "teal"),
        stat("8 met, 4 partly", "of 12 expectations registered before the runs"),
    ], "Most of the narrowing at $10B comes from a target closer to the best one, which we have not explained. "
       "There the agent is statistically indistinguishable from the myopic policy, which knows μ, Σ and the costs (−0.5, SE 0.54)."),
    tag=POST, kicker="Bonus extension 2 · results"))

# 13 ---------------------------------------------------------- conclusions
slides.append(slide(
    "Conclusions",
    f'''<div class="conclusion">
<div class="col3">
<h3>What it learned</h3>
<p>The right assets and a near-optimal mix, from rewards alone. It follows the optimum toward bonds as γ rises.</p>
{stat("14.1 / 19.8", "bps/yr from the best grid portfolio / the optimum (median, γ = 5)", "teal")}
</div>
<div class="col3">
<h3>What stopped it</h3>
<p>Reward noise about 370 times the difference between the top two portfolios. Finer grids and richer models move the error around at this budget.</p>
{stat("3 of 10", "seeds exactly right at γ = 5; 5 of 10 at γ = 3 and 8")}
</div>
<div class="col3">
<h3>What it rests on</h3>
<p>A stationary Gaussian world, μ and Σ taken as true, 260M simulated days per seed. Only the four baseline experiments were pre-registered.</p>
{stat("310.9", "bps/yr lost by equal weight, for scale", "grey")}
</div>
</div>
<p class="note">Post hoc: the 10% grid, the episode-length, budget and exploration variants, function approximation, transaction costs. Benchmark information entered design choices (episode budget; the transaction-cost universe, cost levels and features) and never reached the agent.</p>'''))

# backup --------------------------------------------------------------------
slides.append(slide(
    "Backup: gaps in all four pre-registered experiments",
    wide("fig_gaps", [
        stat("5.74 + 14.1", "20% grid, γ = 5: discretization + median learning gap"),
        stat("11.74 + 2.5", "25% grid, γ = 5"),
        stat("30.6 vs 10.0", "γ = 3: mean vs median learning gap; four seeds are far behind"),
    ]), kicker="Backup"))

slides.append(slide(
    "Backup: where the transaction-cost agent heads, and how often it holds",
    wide("fig_tc_targets_hold", [
        stat("18% to 32%", "IEF in every favorite target at zero, $100M, $1B; the best target holds 14%", "orange"),
        stat("0.17 vs 0.45", "share of blocks with no trade at $10B, agent vs oracle"),
        stat("2.11% vs 1.55%", "turnover per block at $10B, agent vs oracle"),
    ], "Why the target error shrinks at $10B has not been investigated."),
    tag=POST, kicker="Backup"))

slides.append(slide(
    "Backup: it learns the quadratic cost and not the linear one",
    wide("fig_tc_costs", [
        stat("10 of 10", "seeds within a factor of 2 on the quadratic coefficient at $1B and $10B", "teal"),
        stat("10 of 10", "seeds at zero or below 0.2 of the truth on the linear coefficient at $10B", "orange"),
        stat("−1 to −8", "predicted agent − oracle; observed −13.1, −13.1, −10.0, −2.8"),
    ]), tag=POST, kicker="Backup"))

CSS = '''
:root { --navy:#1d3557; --blue:#457b9d; --teal:#2a9d8f; --orange:#e76f51; --grey:#8d99ae; --ink:#22303f; --light:#eef1f6; }
* { box-sizing: border-box; }
html, body { margin:0; height:100%; background:#10151c; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; color:var(--ink); }
#deck { position:fixed; inset:0; display:flex; align-items:center; justify-content:center; overflow:hidden; }
.slide { position:absolute; width:1280px; height:720px; background:#fff; padding:44px 56px 34px; display:none; flex-direction:column; transform-origin:center center; overflow:hidden; }
.slide.active { display:flex; }
.slide h2 { font-size:33px; line-height:1.15; margin:0 0 16px; color:var(--navy); font-weight:700; letter-spacing:-0.3px; max-width:1090px; }
.slide h1 { font-size:46px; line-height:1.1; margin:6px 0 22px; color:var(--navy); letter-spacing:-0.6px; max-width:1100px; }
.kicker { font-size:14px; text-transform:uppercase; letter-spacing:1.4px; color:var(--blue); font-weight:600; margin-bottom:8px; }
.tag { position:absolute; top:40px; right:56px; background:var(--orange); color:#fff; font-size:14px; font-weight:700; padding:5px 12px; border-radius:14px; text-transform:uppercase; letter-spacing:1px; }
.wide { flex:1; display:flex; flex-direction:column; min-height:0; }
.wide .figure { flex:1; min-height:0; display:flex; align-items:center; justify-content:center; }
.figure img { max-width:100%; max-height:100%; object-fit:contain; display:block; }
.stats { display:flex; gap:18px; margin-top:14px; }
.stats.col { flex-direction:column; }
.stat { flex:1; background:var(--light); border-left:6px solid var(--navy); padding:10px 16px 11px; border-radius:3px; }
.stat.teal { border-color:var(--teal); } .stat.orange { border-color:var(--orange); } .stat.grey { border-color:var(--grey); }
.stat .num { font-size:31px; font-weight:700; color:var(--navy); line-height:1.1; white-space:nowrap; }
.stat .lab { font-size:15.5px; line-height:1.25; margin-top:3px; color:#3b4757; }
.split { flex:1; display:flex; gap:34px; min-height:0; align-items:stretch; }
.split .text { flex:0 0 44%; display:flex; flex-direction:column; justify-content:space-between; }
.split.fig-big .text { flex-basis:36%; }
.split.wide-fig .text { flex-basis:33%; justify-content:flex-start; gap:14px; }
.split.fig-big .stats { flex-direction:column; gap:10px; }
.split .figure { flex:1; min-width:0; display:flex; align-items:center; justify-content:center; }
.split .stats .stat .num { font-size:27px; }
.split .stats .stat .lab { font-size:14px; }
ul { margin:0; padding-left:22px; font-size:20px; line-height:1.34; }
li { margin-bottom:9px; }
.lead { font-size:23px; line-height:1.35; margin:0 0 20px; }
.note { font-size:16px; color:#4a5666; margin:12px 0 0; line-height:1.35; }
.conclusion { flex:1; display:flex; gap:30px; align-items:flex-start; padding-top:18px; }
.col3 { flex:1; display:flex; flex-direction:column; }
.col3 h3 { font-size:27px; color:var(--teal); margin:8px 0 8px; }
.col3 p { font-size:25px; line-height:1.4; margin:0 0 22px; min-height:255px; }
.col3 .stat { flex:0 0 auto; min-height:118px; }
.col3 .stat .num { font-size:40px; }
.col3 .stat .lab { font-size:17px; }
.pageno { position:absolute; right:56px; bottom:14px; font-size:13px; color:#9aa4b2; }
#hint { position:fixed; bottom:8px; left:12px; color:#6f7b8a; font-size:12px; }
@page { size: 13.333in 7.5in; margin:0; }
@media print {
  html, body { background:#fff; height:auto; }
  #deck { position:static; display:block; }
  .slide { display:flex !important; position:relative; transform:none !important; page-break-after:always; break-after:page; width:13.333in; height:7.5in; zoom:1; }
  .slide:last-child { page-break-after:auto; break-after:auto; }
  #hint { display:none; }
}
'''

JS = '''
const slides = Array.from(document.querySelectorAll('.slide'));
let cur = 0;
function fit() {
  const s = Math.min(window.innerWidth / 1280, window.innerHeight / 720);
  slides.forEach(el => el.style.transform = 'scale(' + s + ')');
}
function show(i) {
  cur = Math.max(0, Math.min(slides.length - 1, i));
  slides.forEach((el, k) => el.classList.toggle('active', k === cur));
  history.replaceState(null, '', '#' + (cur + 1));
}
document.addEventListener('keydown', e => {
  if (['ArrowRight', 'PageDown', ' ', 'Enter'].includes(e.key)) { show(cur + 1); e.preventDefault(); }
  if (['ArrowLeft', 'PageUp', 'Backspace'].includes(e.key)) { show(cur - 1); e.preventDefault(); }
  if (e.key === 'Home') show(0);
  if (e.key === 'End') show(slides.length - 1);
});
document.addEventListener('click', e => show(cur + (e.clientX < window.innerWidth * 0.3 ? -1 : 1)));
window.addEventListener('resize', fit);
if (!window.matchMedia('print').matches) fit();
window.addEventListener('beforeprint', () => slides.forEach(el => el.style.transform = ''));
window.addEventListener('afterprint', fit);
show(parseInt(location.hash.slice(1) || '1', 10) - 1);
'''

numbered = []
for k, s in enumerate(slides, 1):
    numbered.append(s.replace("</section>", f'<div class="pageno">{k}</div></section>'))

html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model-Free Q-Learning and Optimal Portfolio Choice with Nine Assets</title>
<style>{CSS}</style></head>
<body><div id="deck">
{"".join(numbered)}
</div><div id="hint">← → or click to navigate</div>
<script>{JS}</script></body></html>
'''
(HERE / "slides.html").write_text(html, encoding="utf-8")
print(f"wrote slides.html with {len(slides)} slides, {len(html) / 1e6:.1f} MB")
