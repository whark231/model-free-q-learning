"""Build reports/<run_id>/report.html: one run's results plus a comparison
with every earlier run in reports/registry.json.

Inputs are only committed files: the registry and each run's
reports/<run_id>/data.json (written once per run by src.reporting.extract).
No training logs are needed, so anyone with the repo can rebuild any report.

Run from the repo root:
    python -m src.reporting.build <run_id>
    python -m src.reporting.build --all      # rebuild every registered run
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from src.reporting.extract import (REPORTS, ROOT, SCHEMA_VERSION, canonical_digests, load_registry,
                                  registry_entry)

TEMPLATE = Path(__file__).resolve().parent / "template.html"
PLACEHOLDER = "/*__REPORT_DATA__*/null"

# Parameters a comparison reports as "changed", in display order, with labels.
COMPARED_PARAMS = [
    ("episode_length", "Episode length B (days)"),
    ("n_episodes", "Training episodes per seed"),
    ("simulated_days_per_seed", "Simulated days per seed"),
    ("increment", "Grid step"),
    ("n_actions", "Portfolios on the grid"),
    ("gamma", "Risk aversion γ"),
    ("n_runs", "Seeds"),
    ("master_seed", "Master seed"),
    ("epsilon0", "Initial ε"),
    ("epsilon_at_half", "ε at the halfway point"),
    ("epsilon_final", "ε at the end"),
    ("random_tries_per_portfolio", "Random exploration tries per portfolio"),
    ("alpha_rule", "Learning rate rule"),
    ("agent", "Agent"),
    ("feature_map", "Feature map (function approximation)"),
    ("l2", "l2 penalty"),
    ("greedy_refresh_every", "Greedy refresh (updates)"),
    ("q_init", "Q initialization"),
    ("discount", "Discount ρ"),
    ("log_every", "Checkpoint interval (episodes)"),
    ("convergence_k", "Convergence window K (episodes)"),
    ("near_optimal_bps", "Near-optimal line (bps/yr)"),
    ("dq_tol", "|ΔQ| tolerance"),
    ("n_eval_episodes", "Evaluation episodes"),
]


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(a)), abs(float(b)))
        except (TypeError, ValueError):
            return a == b
    return a == b


def _digests(exp: dict, key: str, prefix: str = "") -> dict:
    commit = exp.get("code", {}).get("git_commit")
    stored = exp.get(key, {}) if key == "calibration_sha256" else exp.get("code", {}).get(key, {})
    return canonical_digests(stored, prefix, commit)


def param_changes(old: dict, new: dict) -> list[dict]:
    """Every compared parameter that differs between two experiments, plus
    changes in calibration data or training code. Files are compared only when
    both runs hashed them (tabular runs do not hash the FA files, audit M3), and
    line endings are ignored (M2)."""
    out = [{"param": k, "label": label, "from": old["params"].get(k), "to": new["params"].get(k)}
           for k, label in COMPARED_PARAMS if not _same(old["params"].get(k), new["params"].get(k))]
    old_cal, new_cal = _digests(old, "calibration_sha256", "data/processed/"), _digests(new, "calibration_sha256", "data/processed/")
    if any(old_cal[f] != new_cal[f] for f in set(old_cal) & set(new_cal)):
        out.append({"param": "calibration", "label": "Calibration data (μ, Σ)", "from": "earlier files", "to": "different files"})
    old_code, new_code = _digests(old, "code_sha256"), _digests(new, "code_sha256")
    changed_files = sorted(f for f in set(old_code) & set(new_code) if old_code[f] != new_code[f])
    if changed_files:
        out.append({"param": "code", "label": "Training code changed", "from": "", "to": ", ".join(changed_files)})
    return out


def comparison_runs(entry: dict, registry: dict) -> list[str]:
    """Run ids to compare against: the entry's explicit `compare_with`, else
    every run registered before it."""
    if entry.get("compare_with") is not None:
        return [r for r in entry["compare_with"] if r != entry["id"]]
    ids = [r["id"] for r in registry["runs"]]
    return ids[: ids.index(entry["id"])]


# Runs extracted before the agent fields existed (2026-11-05) were all tabular.
AGENT_PARAM_DEFAULTS = {"agent": "tabular", "feature_map": None, "l2": 0.0, "greedy_refresh_every": 1}


def load_run_data(run_id: str, reports_dir: Path = REPORTS) -> dict:
    """A run's data.json. Older files lack the agent fields; they get the
    tabular defaults, so comparisons do not show changes that never happened."""
    path = reports_dir / run_id / "data.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `python -m src.reporting.extract {run_id}` on the machine "
                                "that holds that run's training logs")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version", 1) < SCHEMA_VERSION:
        # Fields added since are shown as n/a; re-extract on the machine with the logs (audit F-M3).
        print(f"note: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} is schema "
              f"{data.get('schema_version')} (current {SCHEMA_VERSION}); newer fields show as n/a until "
              f"`python -m src.reporting.extract {run_id}` is re-run", file=sys.stderr)
    for exp in data.get("experiments", []):
        for key, value in AGENT_PARAM_DEFAULTS.items():
            exp.get("params", {}).setdefault(key, value)
    return data


def strip_seed_series(run: dict) -> dict:
    """Prior runs keep per-seed numbers but drop per-seed time series, so a
    report stays small however many runs it layers in."""
    run = copy.deepcopy(run)
    for exp in run["experiments"]:
        for s in exp["seeds"]:
            s.pop("series", None)
            s.pop("excursion_losses", None)
    return run


def build_families(runs: list[dict], reference_id: str | None) -> list[dict]:
    """Group experiments across runs by family (same γ and grid), oldest run
    first, and record what changed against the previous entry and against the
    reference run's entry."""
    fams: dict[str, list] = {}
    for run in runs:
        for exp in run["experiments"]:
            fams.setdefault(exp["family"], []).append({"run_id": run["run"]["id"], "experiment": exp["name"], "exp": exp})
    out = []
    for key, entries in fams.items():
        ref = next((e for e in entries if e["run_id"] == reference_id), entries[0])
        rows = []
        for i, e in enumerate(entries):
            rows.append({
                "run_id": e["run_id"], "experiment": e["experiment"],
                "vs_previous": param_changes(entries[i - 1]["exp"], e["exp"]) if i else [],
                "vs_reference": [] if e is ref else param_changes(ref["exp"], e["exp"]),
            })
        out.append({"family": key, "reference": {"run_id": ref["run_id"], "experiment": ref["experiment"]}, "entries": rows})
    return out


# Registry fields that describe a run. data.json freezes them at extraction, so
# the registry's current values replace them at build time (audit M4).
REGISTRY_RUN_FIELDS = ("title", "date", "git_ref", "scale", "what_changed", "pre_registered")


def overlay_registry(run: dict, registry: dict) -> dict:
    entry = next((r for r in registry["runs"] if r["id"] == run["run"]["id"]), None)
    if entry is not None:
        run["run"].update({k: entry[k] for k in REGISTRY_RUN_FIELDS if k in entry})
    return run


def build_report_data(run_id: str, registry: dict, reports_dir: Path = REPORTS) -> dict:
    entry = registry_entry(run_id, registry)
    prior_ids = comparison_runs(entry, registry)
    current = overlay_registry(load_run_data(run_id, reports_dir), registry)
    prior = [overlay_registry(strip_seed_series(load_run_data(r, reports_dir)), registry) for r in prior_ids]
    order = [r["id"] for r in registry["runs"]]
    runs = sorted(prior, key=lambda r: order.index(r["run"]["id"])) + [current]
    reference = registry.get("reference_run")
    return {
        "current_id": run_id,
        "reference_id": reference if reference in [r["run"]["id"] for r in runs] else None,
        "registry": {r["id"]: r for r in registry["runs"] if r["id"] in [x["run"]["id"] for x in runs]},
        "run_order": [r["run"]["id"] for r in runs],
        "color_slot": {rid: order.index(rid) for rid in [r["run"]["id"] for r in runs]},
        "runs": runs,
        "families": build_families(runs, reference),
        "compared_params": [{"key": k, "label": label} for k, label in COMPARED_PARAMS],
    }


def write_report(run_id: str, registry: dict, reports_dir: Path = REPORTS, template: Path = TEMPLATE) -> Path:
    data = build_report_data(run_id, registry, reports_dir)
    html = template.read_text(encoding="utf-8")
    if html.count(PLACEHOLDER) != 1:
        raise ValueError("template.html must contain the data placeholder exactly once")
    # Escape every "<" so no registry text can close the script or open an HTML comment inside it.
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    out = reports_dir / run_id / "report.html"
    out.write_text(html.replace(PLACEHOLDER, payload), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("run_id", nargs="?")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    registry = load_registry()
    ids = [r["id"] for r in registry["runs"]] if args.all else [args.run_id]
    for rid in ids:
        out = write_report(rid, registry)
        print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
