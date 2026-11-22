"""TC-drift HTML report: built from the committed final-run files only."""

import json

import pytest

from src.tc_drift import report

pytestmark = pytest.mark.skipif(not (report.RUNS / "aum_1b" / "summary.json").exists(),
                                reason="final TC-drift runs not present")


def test_report_builds_deterministically_with_lf(tmp_path):
    a = report.build(out=tmp_path / "a").read_bytes()
    b = report.build(out=tmp_path / "b").read_bytes()
    assert a == b
    assert b"\r\n" not in a
    data = json.loads((tmp_path / "a" / "data.json").read_text())
    assert [e["id"] for e in data["expectations"]] == [f"E{i}" for i in range(1, 13)]


def test_expectation_verdicts_follow_their_clauses():
    for e in report.expectations(report.load()):
        met = [p["met"] for p in e["parts"]]
        assert e["verdict"] == ("met" if all(met) else "not met" if not any(met) else "partly met")


def test_committed_report_matches_a_rebuild(tmp_path):
    committed = report.OUT / "report.html"
    if not committed.exists():
        pytest.skip("report not built")
    assert report.build(out=tmp_path).read_bytes() == committed.read_bytes()
