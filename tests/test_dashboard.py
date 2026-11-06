"""Runs every dashboard page against the saved outputs. Slow, because it needs a finished pipeline run."""

import re
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.serve import clear_uploaded, run_upload, save_uploaded, uploaded_path
from src.utils import project_path

pytestmark = pytest.mark.slow

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
PAGES = sorted(p.name for p in (DASHBOARD / "views").glob("*.py"))
# The page that lists links can only run through the app, which registers those pages.
DIRECT_PAGES = [p for p in PAGES if p != "home.py"]


@pytest.fixture(autouse=True)
def importable(monkeypatch):
    monkeypatch.syspath_prepend(str(DASHBOARD))
    yield
    sys.modules.pop("common", None)


def run_page(name: str) -> AppTest:
    return AppTest.from_file(str(DASHBOARD / "views" / name), default_timeout=120).run()


def page_text(app: AppTest) -> str:
    parts = [e.value for kind in ("title", "markdown", "caption", "info", "warning", "subheader") for e in getattr(app, kind)]
    return " ".join(str(p) for p in parts)


@pytest.mark.parametrize("name", DIRECT_PAGES)
def test_page_runs_without_an_error(name):
    app = run_page(name)
    assert not app.exception, f"{name}: {app.exception}"


def test_the_app_opens_on_the_overview():
    app = AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=120).run()
    assert not app.exception
    assert app.title[0].value == "FlowCast"
    assert "replay" in page_text(app).lower()
    assert "synthetic" in page_text(app).lower()


@pytest.mark.parametrize("name", [p for p in DIRECT_PAGES if p not in ("data_upload.py", "model_training.py")])
def test_every_analysis_page_opens_with_a_question(name):
    app = run_page(name)
    questions = [m.value for m in app.markdown if m.value.startswith("**") and m.value.endswith("?**")]
    assert questions, f"{name} doesn't state the question it answers"


@pytest.mark.parametrize("name", ["accident_risk.py", "congestion_heatmap.py", "road_comparison.py", "historical_trends.py", "weather_vs_traffic.py", "forecast_visualisation.py", "prediction_confidence.py", "feature_importance.py", "data_quality.py", "live_prediction.py", "model_performance.py", "real_data_check.py"])
def test_analysis_pages_state_a_takeaway(name):
    assert run_page(name).info, f"{name} has no takeaway"


def test_the_accident_page_warns_against_using_it_as_an_alert():
    assert "not as an alert" in page_text(run_page("accident_risk.py"))


def test_there_is_no_pages_folder_for_streamlit_to_auto_detect():
    # A folder named pages makes a fresh server skip st.navigation when opened on a direct link.
    assert not (DASHBOARD / "pages").exists()


def test_an_upload_reaches_the_forecast_pages_and_can_be_removed():
    if uploaded_path().exists():
        pytest.skip("A real upload is in place, and this test would overwrite it")
    fixture = Path(__file__).parent / "fixtures" / "sample_upload.csv"
    forecasts = run_upload(fixture)["predictions"]
    stored = pd.read_parquet(project_path("processed") / "predictions.parquet")
    keys = ["road_id", "issue_ts", "horizon"]
    merged = forecasts.merge(stored[[*keys, "volume_pred"]], on=keys, suffixes=("_new", "_old"))
    changed = merged.loc[(merged["volume_pred_new"] - merged["volume_pred_old"]).abs().idxmax()]
    assert abs(changed["volume_pred_new"] - changed["volume_pred_old"]) > 1

    def shown_volume() -> str:
        app = AppTest.from_file(str(DASHBOARD / "views" / "prediction_interface.py"), default_timeout=120).run()
        app.selectbox[0].set_value(changed["road_id"]).run()
        app.selectbox[1].set_value(changed["issue_ts"].strftime("%Y-%m-%d")).run()
        app.selectbox[2].set_value(changed["issue_ts"].strftime("%H:%M")).run()
        app.selectbox[3].set_value(int(changed["horizon"])).run()
        assert not app.exception
        return app.metric[0].value

    assert shown_volume() == f"{changed['volume_pred_old']:.0f}"
    save_uploaded(forecasts)
    try:
        assert shown_volume() == f"{changed['volume_pred_new']:.0f}"
    finally:
        clear_uploaded()
    assert shown_volume() == f"{changed['volume_pred_old']:.0f}"


def test_code_names_stay_out_of_the_opening_of_pages():
    # Internal identifiers should only appear inside the collapsed technical sections.
    code_name = re.compile(r"regress_then_band|linear_scratch|vol_ratio|speed_ratio|y_volume")
    for name in DIRECT_PAGES:
        app = run_page(name)
        visible = " ".join(str(e.value) for kind in ("title", "info", "subheader") for e in getattr(app, kind))
        assert not code_name.search(visible), name
