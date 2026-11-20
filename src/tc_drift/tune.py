"""Tuning pilot for the drift agent at $1B (William; TC_DRIFT_DESIGN.md §9).

In plain terms: train the agent with each candidate setting on the tuning
seeds, then score each trained agent's greedy policy on fresh simulated paths
and keep the setting with the highest realized discounted net reward. Nothing
from the oracle, mu or Sigma is the criterion; they appear only in the
diagnostics written next to it, which are for the report and are not used to
choose.

    python -m src.tc_drift.tune train --candidate a3e-4 --seed 0     # one run, about 2.5 hours with six in parallel
    python -m src.tc_drift.tune summarize                            # after all runs
    python -m src.tc_drift.tune rescore                              # runs trained before the tail average
    python -m src.tc_drift.tune recheck                              # D12 again, after the 2026-11-20 restatement

Decided (William, 2026-11-19; design §6, §9): recursive least squares on the
cost block after a constant-step burn-in, theta at constant alpha_value,
4 million blocks per seed, epsilon reaching 1% at the halfway point, and the
D12 rule as changed that day. Chosen for this pilot by Claude and
confirmed by William on 2026-11-20: the burn-in ends when epsilon first
reaches 0.5 (cost_rls_after = round(ln 2 / beta)), alpha_cost = alpha_value
during the burn-in, target_refresh_every = 1,000 and l2 = 0. Candidates
alpha_value in {3e-4, 1e-3}; 3e-4 won (design §9).

Choosing rule (design §9, fixed before running): the highest mean over 3
tuning seeds wins; if the runner-up is within one paired standard error,
add 3 seeds and decide on 6.

The policy that is scored (changed 2026-11-19 by William, after the first
RLS pilot failed D12 on every run; design §6): greedy on the mean of the
theta snapshots taken at the checkpoints in the final 10% of training (the
tail average), with the costs the agent ended with. The last-iterate policy
is scored too and reported alongside. `rescore` applies this to runs trained
before the change, and marks them post hoc.

How the training log gets theta at every checkpoint: train() records the
costs and the probe actions but not theta (AUDIT_2026-11-06.md TC-6), so the
4 million blocks are run as consecutive train() calls of SEGMENT blocks, a
multiple of the restart interval, with theta read between calls. A call
starts by drawing a start state, which is exactly what one long call does at
every multiple of restart_every, and the return and start generators carry
over, so the segmented run is the same run (tested in
src/tests/test_tc_drift_tune.py).
"""

import argparse
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from src.environment import SeedPlan
from src.hashing import file_sha256
from src.tc_drift import config, env, evaluate, spec, train
from src.tc_drift.agent import DriftAgent
from src.tc_drift.calibration import cost_settings

OUT_DIR = Path("results/tc-drift/tuning")
LEVEL = config.HEADLINE_LEVEL
N_BLOCKS = 4_000_000
SEGMENT = 40_050                          # 445 x RESTART_EVERY; 99 full segments and one of 35,050
BETA = math.log(100) / (N_BLOCKS / 2)     # epsilon = 1% at the halfway point
RLS_AFTER = round(math.log(2) / BETA)     # epsilon = 0.5: the end of the burn-in
CANDIDATES = {
    "a3e-4": {"alpha_value": 3e-4, "alpha_cost": 3e-4},
    "a1e-3": {"alpha_value": 1e-3, "alpha_cost": 1e-3},
}
SHARED = {"beta": BETA, "target_refresh_every": 1000, "l2": 0.0, "cost_rls_after": RLS_AFTER}
FIRST_SEEDS = (0, 1, 2)
EXTRA_SEEDS = (3, 4, 5)
AVERAGE_FRACTION = 0.10                   # the tail average and the D12 window both cover the final 10%
POLICY = "tail_average"
REPO = Path(__file__).resolve().parents[2]
CODE_FILES = ("src/tc_drift/tune.py", "src/tc_drift/agent.py", "src/tc_drift/features.py", "src/tc_drift/train.py",
              "src/tc_drift/env.py", "src/tc_drift/evaluate.py", "src/tc_drift/config.py", "src/tc_drift/spec.py",
              "src/tc_drift/calibration.py", "src/environment.py")


def hyperparameters(candidate: str) -> dict:
    return {**CANDIDATES[candidate], **SHARED}


def segments(n_blocks: int, segment: int) -> list[int]:
    if segment % config.RESTART_EVERY:
        raise ValueError("segments must be a multiple of the restart interval, or the segmented run changes")
    full, rest = divmod(n_blocks, segment)
    return [segment] * full + ([rest] if rest else [])


def train_segmented(agent, market, settings, n_blocks: int, segment: int, returns_rng, start_rng,
                    probes: np.ndarray) -> dict:
    """train() in consecutive calls of `segment` blocks, recording theta, the
    costs and the favourite target after each call."""
    rows = {"checkpoint_block": [], "theta": [], "c1": [], "c2": [], "favourite_target": [], "epsilon": [],
            "probe_greedy": [], "hold_fraction": [], "mean_cost_bps": [], "mean_net_reward": []}
    done = 0
    started = time.perf_counter()
    for length in segments(n_blocks, segment):
        if done:
            elapsed = time.perf_counter() - started
            print(f"progress {done}/{n_blocks} blocks, {elapsed / 60:.0f} min, "
                  f"about {elapsed * (n_blocks - done) / done / 60:.0f} min left", flush=True)
        log = train.train(agent, market, settings, n_blocks=length, returns_rng=returns_rng, start_rng=start_rng,
                          start_states=config.train_start_states(), restart_every=config.RESTART_EVERY,
                          checkpoint_every=length, probe_states=probes)
        done += length
        rows["checkpoint_block"].append(done)
        rows["theta"].append(agent.theta)
        rows["c1"].append(agent.cost_linear)
        rows["c2"].append(agent.cost_quadratic)
        rows["favourite_target"].append(agent.favourite_target())
        rows["epsilon"].append(agent.epsilon)
        rows["probe_greedy"].append(log["probe_greedy"][-1].astype(np.int16))
        for key in ("hold_fraction", "mean_cost_bps", "mean_net_reward"):
            rows[key].append(float(log[key][-1]))
    return {k: np.asarray(v) for k, v in rows.items()}


ONE_STEP = 0.04 + 1e-12      # L1 distance of one 2% step on the grid (two weights move by 0.02)
STABLE_SHARE = 0.95


def post_decision(greedy: np.ndarray, probes: np.ndarray) -> np.ndarray:
    """(n_checkpoints, n_probes, 3): the holdings after each probe's greedy
    action, the target traded to, or the probe itself on HOLD."""
    greedy = np.asarray(greedy)
    out = np.broadcast_to(np.asarray(probes, dtype=float), greedy.shape + (3,)).copy()
    trade = greedy != spec.HOLD
    out[trade] = spec.target_grid()[greedy[trade]]
    return out


def convergence(log: dict, n_blocks: int, probes: np.ndarray | None = None,
                final_fraction: float = 0.10) -> dict:
    """D12 (design §6, as restated 2026-11-20), on the checkpoints of `log`.

    Primary: every checkpoint's favourite target in the final 10% lies within
    L1 0.04 (one 2% step) of the final checkpoint's. Secondary: at least 95%
    of the probe states, the same ones throughout the window, have their
    post-decision holdings (the target traded to, or the probe on HOLD)
    within L1 0.04 of the final checkpoint's at every checkpoint. Reported
    alongside: the 2026-11-19 secondary (95% of probes keep the exact action)
    and the original rule (every probe keeps the exact action).

    The window starts at the last checkpoint at or before 90% of training (the
    first checkpoint if none is), as in evaluate.probe_convergence, and a
    window of one checkpoint passes nothing. For Part IV, a converged run
    reports the first checkpoint from which both rules hold through the end of
    training, in blocks, episodes and daily observations; it is measured back
    from the final policy, so it is not a stopping rule. On a run that has not
    converged those are None, and rules_hold_from_block gives where the rules
    start to hold (inside the window, or None if only the last checkpoint).

    The probes are mostly states far from the no-trade region, where trading
    is right, so the secondary rule barely tests the hold-or-trade decision;
    hold_probes_in_window and hold_decision_stable_share report it on the
    probes that hold at any checkpoint of the window (diagnostics, not rules)."""
    blocks = np.asarray(log["checkpoint_block"])
    first = window_start(blocks, n_blocks, final_fraction)
    targets = spec.target_grid()
    fav_all = np.asarray(log["favourite_target"])
    fav_near = np.abs(targets[fav_all] - targets[fav_all[-1]]).sum(axis=1)
    fav, l1 = fav_all[first:], fav_near[first:]
    greedy_all = np.asarray(log["probe_greedy"])
    greedy = greedy_all[first:]
    kept_every = float((greedy == greedy[-1]).all(axis=0).mean())
    enough = len(fav) > 1
    out = {"window_first_block": int(blocks[first]), "n_checkpoints": int(len(fav)),
           "primary_favourite_within_one_step": bool(enough and (l1 <= ONE_STEP).all()),
           "favourite_max_l1_from_final": float(l1.max()),
           "favourite_distinct": int(len(set(fav.tolist()))),
           "secondary_exact_action_95pct_2026_11_19": bool(enough and kept_every >= STABLE_SHARE),
           "original_all_probes_unchanged": bool(enough and kept_every == 1.0),
           "probes_unchanged_share": kept_every}
    if probes is None:
        return out

    post = post_decision(greedy_all, probes)
    near = np.abs(post - post[-1]).sum(axis=2) <= ONE_STEP                     # (checkpoints, probes)
    within = float(near[first:].all(axis=0).mean())
    out["secondary_95pct_probes_within_one_step"] = bool(enough and within >= STABLE_SHARE)
    out["probes_within_one_step_share"] = within
    out["converged"] = out["primary_favourite_within_one_step"] and out["secondary_95pct_probes_within_one_step"]

    # Part IV: holds[k] says both rules hold over checkpoints k..end (suffixes, so
    # holds is monotone in k); the last checkpoint alone passes nothing.
    fav_suffix = np.flip(np.logical_and.accumulate(np.flip(fav_near <= ONE_STEP)))
    probe_suffix = np.flip(np.logical_and.accumulate(np.flip(near, axis=0), axis=0), axis=0).mean(axis=1)
    holds = fav_suffix & (probe_suffix >= STABLE_SHARE)
    holds[-1] = False
    k = int(np.argmax(holds)) if holds.any() else None
    out["rules_hold_from_block"] = None if k is None else int(blocks[k])
    k = k if out["converged"] else None
    out["converged_from_block"] = None if k is None else int(blocks[k])
    out["converged_from_episode"] = None if k is None else int(blocks[k]) // config.RESTART_EVERY
    out["converged_from_daily_observations"] = None if k is None else int(blocks[k]) * spec.BLOCK_LENGTH

    is_hold = greedy == spec.HOLD
    held = is_hold.any(axis=0)
    out["hold_probes_final"] = int(is_hold[-1].sum())
    out["hold_probes_in_window"] = int(held.sum())
    out["hold_decision_stable_share"] = (float((is_hold[:, held] == is_hold[-1, held]).all(axis=0).mean())
                                         if held.any() else None)
    return out


def window_start(blocks: np.ndarray, n_blocks: int, final_fraction: float = AVERAGE_FRACTION) -> int:
    """Index of the last checkpoint at or before (1 - final_fraction) of
    training, or 0 if there is none."""
    before = np.flatnonzero(np.asarray(blocks) <= (1 - final_fraction) * n_blocks)
    return int(before[-1]) if before.size else 0


def trailing_average(blocks: np.ndarray, theta: np.ndarray, n_blocks: int, end: int,
                     fraction: float = AVERAGE_FRACTION) -> np.ndarray:
    """Mean of the theta snapshots at the checkpoints in the trailing
    `fraction` of training that ends at checkpoint `end`: those with block in
    (blocks[end] - fraction * n_blocks, blocks[end]]. At the last checkpoint
    this is the tail average the scored policy uses."""
    blocks = np.asarray(blocks)
    keep = (blocks > blocks[end] - fraction * n_blocks) & (blocks <= blocks[end])
    return np.asarray(theta)[keep].mean(axis=0)


def averaged_policy(agent: DriftAgent, log: dict, n_blocks: int, probes: np.ndarray) -> tuple:
    """(the tail-averaged agent, D12 for the averaged policy, its favourite
    target at each window checkpoint).

    D12 is applied to the policy we would report had training stopped at
    each checkpoint: the trailing 10% average of theta there, with the costs
    logged there. Every checkpoint is scored, so the Part IV "converged from"
    block can be found; the verdicts read the final-10% window. Consecutive
    trailing averages share most of their snapshots, so they move less than
    the raw iterates by construction; the split-half check compares the
    averages of the two halves of the final 10%, which share no snapshot."""
    blocks = np.asarray(log["checkpoint_block"])
    theta = np.asarray(log["theta"])
    cost = np.column_stack([log["c1"], log["c2"]])
    favourite, greedy = [], []
    for k in range(len(blocks)):
        a = agent.with_parameters(trailing_average(blocks, theta, n_blocks, k), cost[k])
        favourite.append(a.favourite_target())
        greedy.append([a.greedy_action(x) for x in probes])
    trailing = {"checkpoint_block": blocks, "favourite_target": np.array(favourite),
                "probe_greedy": np.array(greedy)}
    conv = convergence(trailing, n_blocks, probes)
    final = agent.with_parameters(trailing_average(blocks, theta, n_blocks, len(blocks) - 1), cost[-1])

    tail = np.flatnonzero(blocks > (1 - AVERAGE_FRACTION) * n_blocks)
    if len(tail) >= 2:
        half = len(tail) // 2
        early = agent.with_parameters(theta[tail[:half]].mean(axis=0), cost[tail[half - 1]])
        late = agent.with_parameters(theta[tail[half:]].mean(axis=0), cost[tail[-1]])
        targets = spec.target_grid()
        conv["split_half_favourite_l1"] = float(np.abs(targets[early.favourite_target()]
                                                       - targets[late.favourite_target()]).sum())
        conv["split_half_favourite_within_one_step"] = bool(conv["split_half_favourite_l1"] <= 0.04 + 1e-12)
        conv["split_half_sizes"] = [int(half), int(len(tail) - half)]
        conv["split_half_probe_agreement"] = float(np.mean([early.greedy_action(x) == late.greedy_action(x)
                                                            for x in probes]))
        halves = post_decision([[early.greedy_action(x) for x in probes], [late.greedy_action(x) for x in probes]],
                               probes)
        conv["split_half_probes_within_one_step"] = float(
            (np.abs(halves[0] - halves[1]).sum(axis=1) <= ONE_STEP).mean())
    conv["n_snapshots_averaged"] = int(len(tail))
    return final, conv, trailing["favourite_target"]


def provenance() -> dict:
    """Commit, whether src/ had uncommitted changes, LF-normalized sha256 of
    every file the run imports for its logic, versions and the seed plan."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "src_dirty": bool(git("status", "--porcelain", "--", "src")),
            "sha256": {f: file_sha256(REPO / f) for f in CODE_FILES},
            "python": platform.python_version(), "numpy": np.__version__,
            "seed_plan": SeedPlan(spec.TUNING_MASTER_SEED, config.N_SEEDS).as_dict(),
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def run_one(candidate: str, seed: int, out_dir: Path = OUT_DIR, n_blocks: int = N_BLOCKS,
            segment: int = SEGMENT, m: int = config.M_EVAL_PILOTS) -> dict:   # the pilot ran at m = 1,000
    market = env.load_market()
    settings = cost_settings()[LEVEL]
    hp = hyperparameters(candidate)
    returns_rng, agent_rng, start_rng = config.run_rngs(seed, spec.TUNING_MASTER_SEED)
    agent = DriftAgent(agent_rng, **hp)
    t0 = time.perf_counter()
    log = train_segmented(agent, market, settings, n_blocks, segment, returns_rng, start_rng, config.probe_states())
    train_seconds = time.perf_counter() - t0

    # The criterion: the final greedy policy's realized discounted net reward on
    # the tuning seed's evaluation paths (the same paths for every candidate and seed).
    last = score(agent, market, settings, m)
    averaged, averaged_conv, _ = averaged_policy(agent, log, n_blocks, config.probe_states())
    scored = score(averaged, market, settings, m)
    eval_seconds = time.perf_counter() - t0 - train_seconds

    out_dir = Path(out_dir) / candidate
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"seed{seed}.npz", **log, value=scored["greedy"]["value"],
                        favourite_rebalance_value=scored["favourite"]["value"],
                        last_iterate_value=last["greedy"]["value"],
                        last_iterate_favourite_rebalance_value=last["favourite"]["value"],
                        averaged_theta=averaged.theta)
    if n_blocks != N_BLOCKS:
        print(f"note: n_blocks = {n_blocks}, not the pilot's {N_BLOCKS}; beta and the burn-in stay at the "
              f"{N_BLOCKS}-block schedule, so this is a smoke run, not a pilot run", flush=True)
    record = {
        "candidate": candidate, "seed": seed, "level": LEVEL, "hyperparameters": hp,
        "n_blocks": n_blocks, "segment": segment, "m_eval": m, "horizon": config.H_EVAL,
        "policy": POLICY, "averaged_post_hoc": False,
        **policy_fields(averaged, scored, averaged_conv),
        "last_iterate": policy_fields(agent, last, convergence(log, n_blocks, config.probe_states())),
        "final_c1": agent.cost_linear, "final_c2": agent.cost_quadratic,
        "train_seconds": round(train_seconds, 1), "eval_seconds": round(eval_seconds, 1),
        "provenance": provenance(),
    }
    (out_dir / f"seed{seed}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    return record


def score(agent: DriftAgent, market, settings, m: int) -> dict:
    """The criterion and its reference on the tuning seed's evaluation paths
    (the same paths for every candidate and seed): the agent's greedy policy,
    and always rebalancing to the agent's own favourite target."""
    starts, paths = evaluate.draw_evaluation_set(market, config.evaluation_rng(spec.TUNING_MASTER_SEED),
                                                 config.H_EVAL, m)
    return {"greedy": evaluate.evaluate_policy(evaluate.agent_policy(agent), starts, paths, settings),
            "favourite": evaluate.evaluate_policy(evaluate.always_rebalance_policy(agent.favourite_target()),
                                                  starts, paths, settings)}


def policy_fields(agent: DriftAgent, scored: dict, conv: dict) -> dict:
    greedy, favourite = scored["greedy"], scored["favourite"]
    return {"criterion_mean_value_bps": greedy["mean_value_bps"],
            "hold_fraction": greedy["mean_hold_fraction"],
            "turnover_per_block": float(greedy["turnover_per_block"].mean()),
            "favourite_rebalance_mean_value_bps": favourite["mean_value_bps"],
            "final_favourite_index": agent.favourite_target(),
            "final_favourite_target": spec.target_grid()[agent.favourite_target()].round(2).tolist(),
            "convergence": conv}


def rescore(out_dir: Path = OUT_DIR) -> list:
    """Apply the tail-averaged policy to runs trained before it was adopted:
    rebuild the agent from the saved hyperparameters, theta snapshots and
    costs, score both policies, and rewrite the run's record and arrays with
    the last-iterate results moved under 'last_iterate'. Marked post hoc."""
    market = env.load_market()
    settings = cost_settings()[LEVEL]
    done = []
    for path in sorted(Path(out_dir).glob("*/seed*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("policy") == POLICY:
            continue
        z = dict(np.load(path.with_suffix(".npz")))
        if "averaged_theta" in z or "last_iterate_value" in z:
            # A rescore that stopped between the two writes: popping "value" again
            # would store the averaged values as the last iterate's.
            raise RuntimeError(f"{path.with_suffix('.npz')} is already rescored but {path.name} is not; "
                               "restore both from the backup and rescore again")
        n_blocks = record["n_blocks"]
        base = DriftAgent(np.random.default_rng(0), **record["hyperparameters"])
        agent = base.with_parameters(z["theta"][-1], (record["final_c1"], record["final_c2"]))
        averaged, conv, _ = averaged_policy(agent, z, n_blocks, config.probe_states())
        scored = score(averaged, market, settings, record["m_eval"])
        last_keys = ("criterion_mean_value_bps", "hold_fraction", "turnover_per_block",
                     "favourite_rebalance_mean_value_bps", "final_favourite_index", "final_favourite_target",
                     "convergence")
        last = {k: record.pop(k) for k in last_keys}
        # The rebuilt agent must be the trained one: same favourite and the same
        # greedy action on every probe state as the training log recorded.
        if not agent.favourite_target() == last["final_favourite_index"] == int(z["favourite_target"][-1]):
            raise RuntimeError(f"{path}: the rebuilt agent's favourite target is not the logged one")
        np.testing.assert_array_equal([agent.greedy_action(x) for x in config.probe_states()], z["probe_greedy"][-1])
        record.update({"policy": POLICY, "averaged_post_hoc": True, **policy_fields(averaged, scored, conv),
                       "last_iterate": last, "rescore_provenance": provenance()})
        z["last_iterate_value"] = z.pop("value")
        z["last_iterate_favourite_rebalance_value"] = z.pop("favourite_rebalance_value")
        z.update(value=scored["greedy"]["value"], favourite_rebalance_value=scored["favourite"]["value"],
                 averaged_theta=averaged.theta)
        np.savez_compressed(path.with_suffix(".npz"), **z)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
        done.append(str(path))
        print(f"rescored {path}", flush=True)
    return done


def recheck(out_dir: Path = OUT_DIR) -> list:
    """Recompute D12 for both policies of finished runs from their saved
    arrays, without scoring them again (for a rule restated after the runs,
    labelled as such in design §6). Leaves the criterion values untouched."""
    probes = config.probe_states()
    done = []
    for path in sorted(Path(out_dir).glob("*/seed*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("policy") != POLICY:
            raise ValueError(f"{path} predates the tail-averaged policy; run `rescore` first")
        z = dict(np.load(path.with_suffix(".npz")))
        base = DriftAgent(np.random.default_rng(0), **record["hyperparameters"])
        agent = base.with_parameters(z["theta"][-1], (record["final_c1"], record["final_c2"]))
        averaged, conv, _ = averaged_policy(agent, z, record["n_blocks"], probes)
        if not np.array_equal(averaged.theta, z["averaged_theta"]):
            raise RuntimeError(f"{path}: the tail average does not match the saved one")
        record["convergence"] = conv
        record["last_iterate"]["convergence"] = convergence(z, record["n_blocks"], probes)
        record["recheck_provenance"] = provenance()
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
        done.append(str(path))
        print(f"rechecked {path}", flush=True)
    return done


def _ranked(values: dict, seeds) -> tuple:
    means = {c: float(np.mean([values[c][s] for s in seeds])) for c in values}
    order = sorted(means, key=means.get, reverse=True)
    diff = np.array([values[order[0]][s] - values[order[1]][s] for s in seeds])
    return order, means, float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))


def choose(values: dict) -> dict:
    """values[candidate][seed] = criterion (bps/yr). Design §9, as
    src/tune_fa.py applies it: every candidate must have FIRST_SEEDS; on those,
    the highest mean wins unless the runner-up is within one paired SE (over
    seeds) of the leader. Then the top two need EXTRA_SEEDS too ('pending'
    until they have them), and the decision is the highest mean on all six.
    Other seeds are ignored."""
    missing = {c: sorted(set(FIRST_SEEDS) - set(v)) for c, v in values.items() if set(FIRST_SEEDS) - set(v)}
    if missing:
        return {"decision": "pending", "missing_first_seeds": missing}
    order, means, diff, se = _ranked(values, FIRST_SEEDS)
    out = {"seeds": list(FIRST_SEEDS), "means": means, "leader": order[0], "runner_up": order[1],
           "paired_difference": diff, "paired_se": se}
    close = diff <= se * (1 + 1e-9) + 1e-12            # tune_fa's tolerance for a tie with the SE
    out["close_call"] = bool(close)
    if not close:
        return {**out, "decision": order[0], "decided_on_seeds": len(FIRST_SEEDS)}
    top = order[:2]
    all_seeds = FIRST_SEEDS + EXTRA_SEEDS
    lacking = {c: sorted(set(all_seeds) - set(values[c])) for c in top if set(all_seeds) - set(values[c])}
    if lacking:
        return {**out, "decision": "pending", "extra_seeds_for": lacking}
    order6, means6, diff6, se6 = _ranked({c: values[c] for c in top}, all_seeds)
    return {**out, "means_6": means6, "paired_difference_6": diff6, "paired_se_6": se6,
            "decision": order6[0], "decided_on_seeds": len(all_seeds)}


def summarize(out_dir: Path = OUT_DIR) -> dict:
    """Collect the runs, apply the choosing rule, and add diagnostics that use
    the benchmark (true costs, the oracle's replayed value, the target's utility
    loss). The diagnostics are reported only; the rule above never reads them."""
    out_dir = Path(out_dir)
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*/seed*.json"))]
    if not records:
        raise FileNotFoundError(f"no runs under {out_dir}")
    for key in ("n_blocks", "segment", "m_eval", "horizon", "level", "policy"):
        if len({json.dumps(r.get(key)) for r in records}) > 1:
            raise ValueError(f"runs under {out_dir} differ in {key}; summarize one pilot at a time")
    for c in {r["candidate"] for r in records}:
        if len({json.dumps(r["hyperparameters"], sort_keys=True) for r in records if r["candidate"] == c}) > 1:
            raise ValueError(f"candidate {c} has runs with different hyperparameters")
    if records[0].get("policy") != POLICY:
        raise ValueError(f"runs under {out_dir} predate the tail-averaged policy; run `rescore` first")
    values, last_values = {}, {}
    for r in records:
        values.setdefault(r["candidate"], {})[r["seed"]] = r["criterion_mean_value_bps"]
        last_values.setdefault(r["candidate"], {})[r["seed"]] = r["last_iterate"]["criterion_mean_value_bps"]
    result = {"post_hoc_runs": sorted(f"{r['candidate']}/seed{r['seed']}" for r in records if r["averaged_post_hoc"]),
              "choice": choose(values) if len(values) > 1 else None,
              "choice_on_last_iterate_for_reference": choose(last_values) if len(values) > 1 else None,
              "runs": records}

    market = env.load_market()
    settings = cost_settings()[LEVEL]
    targets = spec.target_grid()
    U = (targets @ market.mu - 0.5 * settings.gamma * np.einsum("ai,ij,aj->a", targets, market.sigma, targets))
    best = evaluate.best_static_target(market, settings.gamma)
    m = records[0]["m_eval"]
    starts, paths = evaluate.draw_evaluation_set(market, config.evaluation_rng(spec.TUNING_MASTER_SEED),
                                                 config.H_EVAL, m)
    rebalance = evaluate.evaluate_policy(evaluate.always_rebalance_policy(best), starts, paths, settings)
    table_path = Path("results/tc-drift/oracle") / f"{LEVEL}.npz"
    oracle_value = None
    if table_path.exists():
        table = np.load(table_path)["policy_table"]
        oracle = evaluate.evaluate_policy(
            lambda h, t: table[spec.nearest_lattice_index(h, spec.LATTICE_INCREMENT)].astype(np.int64),
            starts, paths, settings)
        oracle_value = oracle["value"]
    c_true = (settings.kappa / spec.BLOCK_LENGTH, settings.lam / (2 * spec.BLOCK_LENGTH))
    diag = {"c_true": c_true, "best_target": targets[best].round(2).tolist(),
            "always_rebalance_mean_value_bps": rebalance["mean_value_bps"], "per_run": []}
    for r in records:
        z = np.load(out_dir / r["candidate"] / f"seed{r['seed']}.npz")
        fav = int(r["final_favourite_index"])
        row = {"candidate": r["candidate"], "seed": r["seed"],
               "c1_over_true": r["final_c1"] / c_true[0], "c2_over_true": r["final_c2"] / c_true[1],
               "favourite_target_utility_loss_bps": float((U[best] - U[fav]) * spec.TRADING_DAYS * spec.BPS),
               "agent_minus_always_rebalance": evaluate.paired_gap(z["value"], rebalance["value"], config.H_EVAL),
               "agent_minus_favourite_rebalance": evaluate.paired_gap(z["value"], z["favourite_rebalance_value"],
                                                                      config.H_EVAL),
               "favourite_rebalance_minus_always_rebalance": evaluate.paired_gap(
                   z["favourite_rebalance_value"], rebalance["value"], config.H_EVAL),
               "averaged_minus_last_iterate": evaluate.paired_gap(z["value"], z["last_iterate_value"],
                                                                  config.H_EVAL)}
        if oracle_value is not None:
            row["agent_minus_oracle"] = evaluate.paired_gap(z["value"], oracle_value, config.H_EVAL)
            row["last_iterate_minus_oracle"] = evaluate.paired_gap(z["last_iterate_value"], oracle_value,
                                                                   config.H_EVAL)
        diag["per_run"].append(row)
    if oracle_value is not None:
        diag["oracle_minus_always_rebalance"] = evaluate.paired_gap(oracle_value, rebalance["value"], config.H_EVAL)
    result["diagnostics_not_used_to_choose"] = diag
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--n-blocks", type=int, default=N_BLOCKS)
    t.add_argument("--out", type=Path, default=OUT_DIR)
    s = sub.add_parser("summarize")
    s.add_argument("--out", type=Path, default=OUT_DIR)
    r = sub.add_parser("rescore")
    r.add_argument("--out", type=Path, default=OUT_DIR)
    c = sub.add_parser("recheck")
    c.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(run_one(args.candidate, args.seed, args.out, args.n_blocks), indent=2), flush=True)
    elif args.command == "rescore":
        rescore(args.out)
    elif args.command == "recheck":
        recheck(args.out)
    else:
        print(json.dumps(summarize(args.out)["choice"], indent=2))
