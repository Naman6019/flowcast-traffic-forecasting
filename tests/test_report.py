"""The final report is written by hand, so these checks catch its key figures drifting from the saved results."""

import json

import pytest

from src.utils import project_path

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def report():
    return (project_path("reports") / "final_report.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def metrics():
    return json.loads((project_path("reports") / "final_metrics.json").read_text(encoding="utf-8"))


def test_headline_figures_match_the_saved_metrics(report, metrics):
    assert f"{metrics['volume']['mape_pct']:.2f}%" in report
    assert f"{metrics['congestion']['macro_f1']:.3f}" in report
    assert f"{metrics['accident']['roc_auc']:.3f}" in report
    assert f"{metrics['coverage']['volume_80']:.1f}%" in report
    assert f"{metrics['benchmark']['deep_test_rmse']:.2f}" in report
    assert f"{metrics['benchmark']['selected_classical_test_rmse']:.2f}" in report
    assert f"{metrics['benchmark']['best_classical_test_rmse']:.2f}" in report


def test_intervals_match_the_saved_metrics(report, metrics):
    low, high = metrics["intervals_95"]["congestion_macro_f1"]
    assert f"{low:.3f} to {high:.3f}" in report
    low, high = metrics["intervals_95"]["accident_roc_auc"]
    assert f"{low:.3f} to {high:.3f}" in report
    low, high = metrics["benchmark"]["gain_interval_95"]
    assert f"{low:.2f} to {high:.2f}" in report


def test_real_data_figures_match_the_saved_results(report):
    results = json.loads((project_path("reports") / "real_data_results.json").read_text(encoding="utf-8"))
    best = results["headline"]
    assert f"{best['test_wape_pct']:.1f}%" in report
    assert f"{best['test_rmse']:.0f}" in report
    assert f"{results['skill_vs_persistence']:.0%}" in report
    assert f"{results['coverage_80']:.1f}%" in report
    low, high = results["wape_interval"]
    assert f"{low:.1f} to {high:.1f}" in report


def test_missed_targets_are_reported_as_missed(report, metrics):
    missed = [c["criterion"] for c in metrics["criteria"] if not c["met"]]
    assert missed == ["Congestion macro-F1 >= 0.80", "Accident ROC-AUC >= 0.75"]
    assert report.count("**Not met**") == len(missed)
