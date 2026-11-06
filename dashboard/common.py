"""Shared loaders, plain-language labels and page furniture. Every number on screen comes from a saved pipeline output."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.clean import load_dataset  # noqa: E402
from src.model_card import read_registry  # noqa: E402
from src.serve import overlay_uploaded, uploaded_path  # noqa: E402
from src.utils import load_config, project_path  # noqa: E402

CONFIG = load_config()
LABELS = CONFIG["congestion"]["labels"]
SEVERITY_COLOURS = {"Free-flow": "#2e9e5b", "Moderate": "#e3b505", "Heavy": "#ee7b30", "Severe": "#c8102e"}
SEVERITY_SEQUENCE = [SEVERITY_COLOURS[label] for label in LABELS]
HORIZON_LABELS = {1: "30 minutes", 2: "60 minutes", 3: "90 minutes", 4: "120 minutes"}
REPORTS = ROOT / "reports"

MODEL_NAMES = {
    "linear_scratch": "Linear regression, written by hand in NumPy",
    "linear": "Linear regression",
    "logistic": "Logistic regression",
    "tree": "Decision tree",
    "forest": "Random forest",
    "xgboost": "XGBoost",
    "svm_rbf": "Support vector machine (RBF)",
    "svm_linear": "Support vector machine (linear)",
    "regress_then_band": "Volume forecast + published congestion rule",
    "regress_then_band_tuned": "Volume forecast + tuned congestion cut-offs",
    "derived_from_speed": "Worked out from forecast speed",
    "lstm": "LSTM neural network",
    "gru": "GRU neural network",
    "bilstm": "Bidirectional LSTM neural network",
    "lstm_multitask": "LSTM with a congestion output",
    "Persistence (last window)": "Simple rule: repeat the latest reading",
    "Same window yesterday": "Simple rule: same time yesterday",
    "Same window last week": "Simple rule: same time last week",
    "Slot profile": "Simple rule: usual level so far",
}
GROUP_NAMES = {
    "Recent volume": "Recent traffic",
    "Recent speed": "Recent speed",
    "Usual level for the slot": "Usual level for this road and time",
    "Today against usual": "How today compares with usual",
    "Clock": "Time of day and week",
    "Calendar": "Holidays, events and roadworks",
    "Weather": "Weather",
    "Segment": "Which road",
    "Mix and operations": "Vehicle mix, signals and recent accidents",
}
TARGET_NAMES = {"volume": "Traffic volume", "speed": "Speed", "travel_time": "Travel time", "congestion": "Congestion level", "accident": "Accident risk"}

_MEASURES = {"vol": "volume", "speed": "speed"}
FEATURE_NAMES = {
    "hour_sin": "Time of day (cycle, part 1)",
    "hour_cos": "Time of day (cycle, part 2)",
    "dow_sin": "Day of week (cycle, part 1)",
    "dow_cos": "Day of week (cycle, part 2)",
    "is_weekend": "Weekend",
    "is_peak": "Peak period",
    "public_holiday": "Public holiday",
    "event_flag": "Local event",
    "roadwork_flag": "Roadworks",
    "holiday_x_peak": "Holiday during peak",
    "event_x_peak": "Event during peak",
    "occ_t": "Road occupancy now",
    "vc_t": "How full the road is now (volume / capacity)",
    "headroom_t": "Spare capacity now",
    "road_capacity": "Road capacity",
    "seg_length_km": "Segment length",
    "road_code": "Which road",
    "wx_rainfall": "Rainfall, last full hour",
    "wx_visibility": "Visibility, last full hour",
    "wx_temperature": "Temperature, last full hour",
    "wx_rain": "Raining, last full hour",
    "wx_low_vis": "Low visibility, last full hour",
    "wx_temp_band": "Temperature band",
    "wx_is_clear": "Clear weather",
    "wx_is_cloudy": "Cloudy weather",
    "wx_is_rain": "Rainy weather",
    "wx_is_fog": "Foggy weather",
    "share_hcv_t": "Share of heavy vehicles now",
    "share_2w_t": "Share of two-wheelers now",
    "signal_timing_t": "Green-light time now",
    "accidents_last4": "Accidents in the last 2 hours",
    "rain_x_peak": "Rain during peak",
    "low_vis_x_speed": "Low visibility and speed combined",
}
for _short, _word in _MEASURES.items():
    FEATURE_NAMES.update(
        {
            f"{_short}_t": f"Latest {_word}",
            f"{_short}_t1": f"{_word.capitalize()} 30 minutes earlier",
            f"{_short}_t2": f"{_word.capitalize()} 60 minutes earlier",
            f"{_short}_day": f"{_word.capitalize()} at this time yesterday",
            f"{_short}_week": f"{_word.capitalize()} at this time last week",
            f"{_short}_mean4": f"Average {_word}, last 2 hours",
            f"{_short}_mean8": f"Average {_word}, last 4 hours",
            f"{_short}_std4": f"How much {_word} varied, last 2 hours",
            f"{_short}_std8": f"How much {_word} varied, last 4 hours",
            f"{_short}_profile": f"Usual {_word} for this road and time",
            f"{_short}_ratio_t": f"{_word.capitalize()} now against usual",
            f"{_short}_ratio_mean4": f"{_word.capitalize()} against usual, last 2 hours",
        }
    )

GLOSSARY = {
    "Forecast window": "Traffic is measured in 30-minute periods. A forecast 30 minutes ahead is for the next period.",
    "Volume": "How many vehicles passed in a 30-minute period.",
    "Congestion level": "Free-flow, Moderate, Heavy or Severe, set by how full the road is compared with its capacity.",
    "Peak": "The busy periods, 07:30 to 10:00 and 17:00 to 20:00.",
    "RMSE": "Typical size of a forecast error, in the measure's own units. Large misses count extra, so lower is better.",
    "MAE": "Average size of a forecast error, in the measure's own units. Lower is better.",
    "MAPE": "Average forecast error as a percentage of the actual value. Lower is better.",
    "WAPE": "Total forecast error as a share of total traffic. Unlike MAPE, quiet hours with small counts can't inflate it.",
    "R²": "How much of the ups and downs the forecast explains. 1.0 would be perfect.",
    "Macro-F1": "A score from 0 to 1 for classifying congestion that counts each level equally, so rare Severe periods matter as much as common Free-flow ones.",
    "ROC-AUC": "If you took one period with an accident and one without, this is how often the model ranks the accident period higher. 0.5 is a coin toss, and 1.0 is perfect.",
    "Validation and test": "The model is tuned on validation dates (16 April to 8 May), and judged on test dates (9 to 31 May) that played no part in choosing it.",
    "80% range": "The range the actual value should fall in 8 times out of 10. It comes from how large forecast errors were on validation dates (a method called conformal prediction).",
    "Reliability": "95th-percentile travel time divided by the median. 1.0 means steady, and higher means a bad day is much worse than a typical one.",
}


def _local(frame: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drop the timezone for display. All times shown are corridor local time."""
    for col in columns:
        frame[col] = frame[col].dt.tz_localize(None)
    return frame


@st.cache_data(show_spinner="Loading the dataset")
def dataset() -> pd.DataFrame:
    return _local(load_dataset(CONFIG), ["ts"])


def _with_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _local(frame, ["issue_ts", "target_ts"])
    frame["congestion_label"] = frame["congestion_pred"].map(dict(enumerate(LABELS)))
    frame["actual_label"] = frame["actual_congestion"].map(dict(enumerate(LABELS)))
    return frame


@st.cache_data(show_spinner="Loading forecasts")
def _stored_predictions() -> pd.DataFrame:
    return pd.read_parquet(project_path("processed", CONFIG) / "predictions.parquet")


@st.cache_data(show_spinner="Loading uploaded forecasts")
def _combined_predictions(modified: float) -> pd.DataFrame:
    return _with_labels(overlay_uploaded(_stored_predictions(), pd.read_parquet(uploaded_path(CONFIG))))


@st.cache_data(show_spinner="Loading forecasts")
def _base_predictions() -> pd.DataFrame:
    return _with_labels(_stored_predictions().assign(source="stored"))


def predictions(include_uploads: bool = True) -> pd.DataFrame:
    """Saved forecasts, with any uploaded ones taking the place of the same road, time and look-ahead.

    Accuracy views pass include_uploads=False, so that they always describe the evaluated models and data.
    """
    path = uploaded_path(CONFIG)
    if include_uploads and path.exists():
        return _combined_predictions(path.stat().st_mtime)
    return _base_predictions()


def has_upload() -> bool:
    return uploaded_path(CONFIG).exists()


@st.cache_data
def segments() -> pd.DataFrame:
    return pd.read_parquet(project_path("processed", CONFIG) / "segment_static.parquet").sort_values("road_id")


@st.cache_data
def report_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / name)


@st.cache_data
def report_json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


@st.cache_data
def registry() -> dict:
    return read_registry(CONFIG)


def road_names() -> dict:
    table = segments()
    return dict(zip(table["road_id"], table["road_name"], strict=True))


def segment_picker(label: str = "Road", key: str | None = None, default: int = 0) -> str:
    """Pick a segment by name, returning its road_id."""
    names = road_names()
    return st.selectbox(label, list(names), index=default, format_func=lambda r: f"{names[r]} ({r})", key=key)


def page_header(title: str, question: str, how_to_read: str | None = None) -> None:
    """Every page opens with the question it answers, then how to read it."""
    st.title(title)
    st.markdown(f"**{question}**")
    if how_to_read:
        st.caption(how_to_read)


def takeaway(text: str) -> None:
    """The one thing to notice on this page, worked out from the data on screen."""
    st.info(text, icon=":material/lightbulb:")


def terms(*names: str) -> None:
    """A collapsed explanation of the technical terms a page uses."""
    with st.expander("What do these terms mean?"):
        for name in names:
            st.markdown(f"**{name}:** {GLOSSARY[name]}")


def default_moment(forecasts: pd.DataFrame) -> pd.Timestamp:
    """A moment worth looking at: 08:00 on the last weekday of the test period, when the morning peak is building."""
    issue = forecasts.loc[(forecasts["split"] == "test") & (forecasts["horizon"] == 4), "issue_ts"]
    morning = issue[(issue.dt.hour == 8) & (issue.dt.minute == 0) & (issue.dt.day_of_week < 5)]
    return morning.max() if not morning.empty else issue.max()


def moment_picker(forecasts: pd.DataFrame) -> tuple:
    """Pick the moment to treat as now and how far ahead to look. The choice carries across pages."""
    default = st.session_state.get("moment", default_moment(forecasts))
    default_horizon = st.session_state.get("horizon", 1)
    days = sorted(forecasts["issue_ts"].dt.strftime("%Y-%m-%d").unique())
    c1, c2, c3 = st.columns(3)
    day_text = default.strftime("%Y-%m-%d")
    day = c1.selectbox("Pretend today is", days, index=days.index(day_text) if day_text in days else len(days) - 1)
    times = sorted(forecasts.loc[forecasts["issue_ts"].dt.strftime("%Y-%m-%d") == day, "issue_ts"].dt.strftime("%H:%M").unique())
    clock_text = default.strftime("%H:%M")
    clock = c2.selectbox("And the time is just after", times, index=times.index(clock_text) if clock_text in times else 0)
    horizons = list(HORIZON_LABELS)
    horizon = c3.selectbox("Look ahead", horizons, index=horizons.index(default_horizon), format_func=HORIZON_LABELS.get)
    now = pd.Timestamp(f"{day} {clock}")
    st.session_state["moment"], st.session_state["horizon"] = now, horizon
    return now, horizon


def model_details(frame: pd.DataFrame, columns: tuple = ("volume_model",)) -> None:
    """Which model versions produced what is on screen (Transparency NFR), out of the way until wanted."""
    versions = sorted({str(v) for col in columns for v in frame[col].unique()})
    with st.expander("Which models produced this?"):
        st.markdown("\n".join(f"- `{v}`" for v in versions))
        st.caption("Each version has its full record on the Model records page.")
