# 08 First transaction-cost design (working notes)

**TL;DR / In plain terms:** the first TC design (no drift, state = previous grid target, quadratic cost) is kept as a negative result. Its trained policy scores like a random grid portfolio. How it appears in the report (a paragraph, or with tables) is the open decision D-G.

Source: `docs/methods/TRANSACTION_COSTS.md` (72 lines); code `src/transaction_costs.py`, `src/tc_sweep_report.py`; results `results/tc-quadratic-cost/{no-cost,low-cost,anchor,high-cost}.json`; report `reports/tc-quadratic-cost/`.

## Numbers quoted (expected net CE, bps/yr, discounted at 0.95 over 100 blocks; learned = median over seeds)
| Cost per 20-pt transfer | Grid oracle | Ignore costs | Myopic | Learned | Learned gap median (range) |
|---|---|---|---|---|---|
| 0 bp | 524.8 | 524.8 | 524.8 | 56.6 | 468.2 (436.4 to 485.2) |
| 1 bp | 520.9 | 520.8 | 515.9 | −16.8 | 537.7 (482.4 to 589.8) |
| 2 bp | 517.1 | 516.8 | 512.6 | −90.9 | 608.0 (572.5 to 658.7) |
| 5 bp | 507.6 | 504.9 | 479.2 | −296.8 | 804.3 (744.9 to 844.3) |

Average grid portfolio 56.7. 1.66M (holding, target) pairs; 57.1% ever tried. Reward SD about 3,900 bps/yr per block. Commit `da2b0fa`.

## Open items (waiting on D-G)
M7 (stability criterion cannot be met), M8 (report text in `tc_sweep_report.py` overstates), M9 (single-variant configs not in repo; doc command writes one file with four variants), M11 (2 tests only), TC halves of M1 (Windows print) and F-L12 (SVG metadata), H2 (`ignore_cost` not shown in the report tables).

## Stale text in the doc
- Key takeaways bullet 2: "`AUDIT_2026-11-06.md` §6.2 lists redesign options. A decision is needed before any further TC run." The decision was made on 2026-11-12 (redesign with drift). The "Superseded" line above it says so, so the bullet contradicts its own header.
- "**Next.** Options are in `AUDIT_2026-11-06.md` §6.2." Same.
- TL;DR last sentence "The design needs to change before the extension can be reported as a success." Superseded by the redesign.

## Unrun configs
`experiments/transaction-costs/discount.json` and `learning-rate.json` are registered, not run, and "should not be run as designed" (audit H1). They are orphan configs unless D-G decides otherwise.
