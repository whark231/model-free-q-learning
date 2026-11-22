# 11 Tests (working notes)

**TL;DR / In plain terms:** the suite passed in full on this Mac on 2026-11-20: 758 tests in 288 seconds. The docs said 757; the extra test arrived with commit `c9de7df`. I corrected the two places that said 757.

Command: `OMP_NUM_THREADS=1 /opt/anaconda3/bin/python -m pytest src/tests -q` → `758 passed in 287.64s`.

| File | Tests | What it covers |
|---|---|---|
| test_data_pipeline.py | 22 | Cleaning outputs, sample, adjustment |
| test_environment.py | 77 | Sampler moments, seeds, stream separation, leakage guard |
| test_mean_variance.py | 29 | Closed form, long-only solver, best grid action |
| test_action_space.py | 36 | Counts 495 / 1,287 / 43,758, weights ≥ 0 and sum to 1, no benchmark inputs |
| test_rewards.py | 9 | Reward equals objective in expectation, ddof |
| test_agent.py | 24 | Update rule, ε schedule, tie-breaks |
| test_train.py | 58 | Loop, checkpoints, restart, convergence rules |
| test_config.py | 46 | Pinned settings and registered runs |
| test_evaluate.py | 35 | Gaps, rules, outputs |
| test_reporting.py | 34 | Extraction and report build |
| test_features.py, test_fa_agent.py, test_tune_fa.py | 26, 25, 25 | FA features, agent, tuning |
| test_transaction_costs.py | 2 | First TC design (M11: thin) |
| test_tc_drift_*.py (9 files) | 310 | Spec, env, calibration, oracle, oracle runs, agent, train/evaluate, run driver, tuning |

Total 758.

## Do the tests test the claims?
- Counts, constraint satisfaction, reward/objective consistency, closed form and seed separation are tested directly and I re-derived each independently.
- Leakage is tested by import allowlists. That catches code paths; it cannot catch human design choices made with benchmark numbers in view (the 13M budget, TC design choices). Those are disclosures, not tests.
- `test_config.py` pins settings, which protects pre-registration from silent edits.
- Thin spot: the first TC design has 2 tests (M11, waits on D-G).
- Not run by me: anything on Windows. Last Windows record is 672 passed on 2026-11-19 (`AUDIT_2026-11-06.md` §9.5).
