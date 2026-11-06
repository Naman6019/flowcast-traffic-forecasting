import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import CONFIG, MODEL_NAMES, REPORTS, page_header, takeaway, terms

from src.utils import project_path

page_header(
    "Real-data check",
    "Does this approach work on real traffic?",
    "The Northline data looks synthetic, so the same approach was also tested on real sensor counts: hourly traffic on westbound I-94 in Minnesota, from a public dataset. The models never saw the test period, January to September 2018, while being built or chosen.",
)

path = REPORTS / "real_data_results.json"
if not path.exists():
    st.info("These results haven't been generated yet. Run `python -m src.pipeline real`.")
    st.stop()

results = json.loads(path.read_text(encoding="utf-8"))
best = results["headline"]
names = {**MODEL_NAMES, "Persistence (last hour)": "Simple rule: repeat the last hour", "Same hour yesterday": "Simple rule: same hour yesterday", "Same hour last week": "Simple rule: same hour last week", "Slot profile": "Simple rule: usual level so far"}
takeaway(
    f"Yes, for traffic volume. One hour ahead, the {names.get(best['model'], best['model'])} was off by {best['test_wape_pct']:.1f}% of traffic on average, and its typical error was {results['skill_vs_persistence']:.0%} smaller than simply repeating the last hour. "
    "This dataset only has traffic counts, so congestion, travel time and accident risk couldn't be checked on real data."
)

a, b, c = st.columns(3)
a.metric("Average error, 1 hour ahead", f"{best['test_wape_pct']:.1f}%", help="Total error as a share of total traffic (WAPE).")
b.metric("Better than repeating the last hour", f"{results['skill_vs_persistence']:.0%}", help="How much smaller the typical error (RMSE) is.")
c.metric("80% range contained the actual value", f"{results['coverage_80']:.1f}%", help="Should be close to 80%.")

st.subheader("Forecast against what happened")
forecasts = pd.read_parquet(project_path("processed", CONFIG) / "real_predictions.parquet")
days = sorted(forecasts["target_ts"].dt.strftime("%Y-%m-%d").unique())
start, end = st.select_slider("Dates", options=days, value=(days[0], days[min(6, len(days) - 1)]))
part = forecasts[forecasts["target_ts"].dt.strftime("%Y-%m-%d").between(start, end)]
figure = go.Figure()
figure.add_scatter(x=part["target_ts"], y=part["high80"], line=dict(width=0), showlegend=False, hoverinfo="skip")
figure.add_scatter(x=part["target_ts"], y=part["low80"], fill="tonexty", fillcolor="rgba(42,111,151,0.25)", line=dict(width=0), name="80% range")
figure.add_scatter(x=part["target_ts"], y=part["forecast"], name="Forecast, 1 hour ahead", line=dict(color="#2a6f97", width=1.5))
figure.add_scatter(x=part["target_ts"], y=part["actual"], name="What happened", line=dict(color="#222", width=1.3))
figure.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Vehicles per hour", legend=dict(orientation="h", y=1.08))
st.plotly_chart(figure, width="stretch")

st.subheader("How it compares with the supplied data")
near = results["northline"]
comparison = pd.DataFrame(
    [
        ("Average error", f"{near['model']['wape_pct']:.1f}%", f"{best['test_wape_pct']:.1f}%"),
        ("Better than repeating the last reading", f"{near['skill_vs_persistence']:.0%}", f"{results['skill_vs_persistence']:.0%}"),
        ("Better than the same time yesterday", f"{near['skill_vs_yesterday']:.0%}", f"{results['skill_vs_yesterday']:.0%}"),
    ],
    columns=["Measure", "Northline (supplied, 30 minutes ahead)", "I-94 (real, 1 hour ahead)"],
)
st.dataframe(comparison, hide_index=True, width="stretch")
st.caption("A guide only. These are different roads, time steps and look-aheads, so the numbers aren't directly comparable.")

st.subheader("Every model, one hour ahead")
board = pd.DataFrame(results["scoreboard"]).sort_values("test_rmse")
board = board.assign(Model=board["model"].map(lambda m: names.get(m, m)))
st.dataframe(
    board[["Model", "val_rmse", "test_rmse", "test_wape_pct", "test_r2"]]
    .rename(columns={"val_rmse": "Validation RMSE", "test_rmse": "Test RMSE", "test_wape_pct": "Test WAPE (%)", "test_r2": "Test R²"})
    .style.format({"Validation RMSE": "{:.1f}", "Test RMSE": "{:.1f}", "Test WAPE (%)": "{:.1f}", "Test R²": "{:.3f}"}),
    hide_index=True,
    width="stretch",
)

st.subheader("Where it struggles")
left, right = st.columns(2)
with left:
    st.markdown("**By weather at the forecast hour**")
    weather = pd.DataFrame(results["by_weather"]).sort_values("n", ascending=False)
    st.dataframe(
        weather[["value", "n", "mape_pct", "bias"]].rename(columns={"value": "Weather", "n": "Hours", "mape_pct": "Average error (%)", "bias": "Too high (+) or low (-)"}).style.format({"Hours": "{:,}", "Average error (%)": "{:.1f}", "Too high (+) or low (-)": "{:+.0f}"}),
        hide_index=True,
        width="stretch",
    )
with right:
    st.markdown("**By look-ahead**")
    horizons = pd.DataFrame(results["by_horizon"])
    st.dataframe(
        horizons.assign(Ahead=horizons["horizon"].map(lambda h: f"{h} hour{'s' if h > 1 else ''}"), Model=horizons["model"].map(lambda m: names.get(m, m)))[["Ahead", "Model", "test_wape_pct", "seasonal_test_wape_pct"]]
        .rename(columns={"Ahead": "Look ahead", "test_wape_pct": "Average error (%)", "seasonal_test_wape_pct": "Same hour yesterday (%)"})
        .style.format({"Average error (%)": "{:.1f}", "Same hour yesterday (%)": "{:.1f}"}),
        hide_index=True,
        width="stretch",
    )

with st.expander("What was wrong with this data"):
    st.dataframe(pd.DataFrame(list(results["cleaning"].items()), columns=["Problem", "How many"]).style.format({"How many": "{:,}"}), hide_index=True, width="stretch")
    st.caption("Short gaps are estimated and never scored. Longer gaps, including one of about ten months, stay blank, and hours with too little history aren't forecast.")
with st.expander("Full write-up"):
    st.markdown((REPORTS / "real_data.md").read_text(encoding="utf-8"))
terms("WAPE", "RMSE", "MAPE", "R²", "80% range", "Validation and test")
