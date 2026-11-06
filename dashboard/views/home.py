import json

import pandas as pd
import streamlit as st
from common import GLOSSARY, REPORTS, default_moment, predictions, report_csv, report_json, road_names

metrics = report_json("final_metrics.json")
criteria = {c["criterion"]: c["met"] for c in metrics["criteria"]}
forecasts = predictions(include_uploads=False)

st.title("FlowCast")
st.markdown("#### Predicts traffic on 25 road segments, 30 to 120 minutes ahead, so that a control room can act before congestion forms.")
st.markdown("For every road, it forecasts how many vehicles to expect, how fast they'll move, how long the trip will take, how congested the road will be, and how likely an accident is. Each forecast comes with a measure of how sure it is.")

st.warning(
    "**This is a replay, not a live system.** It's an internship project built on a supplied dataset covering 1 January to 31 May 2025, which looks synthetic. "
    "It isn't connected to real roads. Forecasts exist for 16 April to 31 May, and on the forecast pages you choose a past moment to treat as now.",
    icon=":material/history:",
)

st.subheader("What do you want to do?")
left, right = st.columns(2)
with left:
    st.markdown("**Run the corridor today**")
    st.page_link("views/congestion_heatmap.py", label="See which roads jam, and when", icon=":material/grid_on:")
    st.page_link("views/live_prediction.py", label="See the next 2 hours on every road", icon=":material/schedule:")
    st.markdown("**Decide where to position patrols**")
    st.page_link("views/accident_risk.py", label="Rank roads by accident risk", icon=":material/emergency:")
with right:
    st.markdown("**Plan road improvements**")
    st.page_link("views/road_comparison.py", label="Compare roads side by side", icon=":material/compare_arrows:")
    st.page_link("views/weather_vs_traffic.py", label="See what rain and fog do to traffic", icon=":material/rainy:")
    st.markdown("**Check whether to trust it**")
    st.page_link("views/model_performance.py", label="See how accurate it is", icon=":material/verified:")
    st.page_link("views/data_quality.py", label="See what was wrong with the data", icon=":material/rule:")

st.subheader("A sample moment")
now = default_moment(forecasts)
sample = forecasts[(forecasts["issue_ts"] == now) & (forecasts["horizon"] == 1)]
top = sample.loc[sample["accident_score"].idxmax()]
st.caption(f"The forecast made at {now:%H:%M on %A %d %B %Y}, for the next 30 minutes.")
a, b, c = st.columns(3)
a.metric("Vehicles across all 25 roads", f"{sample['volume_pred'].sum():,.0f}")
b.metric("Time to drive the whole corridor", f"{sample['travel_time_pred'].sum():.0f} min")
c.metric("Roads in Heavy or Severe congestion", f"{int(sample['congestion_label'].isin(['Heavy', 'Severe']).sum())} of 25")
st.markdown(f"It's the morning peak. The road ranked highest for accident risk is **{road_names()[top['road_id']]}**, at a {top['accident_prob']:.1%} chance.")

st.subheader("How well does it work?")
st.caption("Measured on 9 to 31 May, dates that played no part in building or choosing the models. Forecasts are for 30 minutes ahead.")
persistence = report_csv("scoreboard_volume.csv").set_index("model").loc["Persistence (last window)", "test_mape_pct"]
accident = metrics["accident"]
rows = [
    ("Traffic volume", f"Off by {metrics['volume']['mape_pct']:.1f}% on average. Just repeating the latest reading would be off by {persistence:.1f}%.", "At most 12% off", "Met" if criteria["Volume MAPE <= 12%"] else "Missed"),
    (
        "Congestion level",
        f"Exactly right {metrics['congestion']['accuracy']:.0%} of the time, and mistakes are nearly always one level off. Harder for Heavy and Severe, which are rare (F1 score {metrics['congestion']['macro_f1']:.3f}).",
        "F1 score of at least 0.80",
        "Met" if criteria["Congestion macro-F1 >= 0.80"] else "Missed",
    ),
    (
        "Accident risk",
        f"A weak signal. Between one period with an accident and one without, it ranks the accident period higher {accident['roc_auc']:.0%} of the time. Its top 10% of periods see {accident['top_decile_lift']:.1f} times the usual accident rate.",
        "At least 75%",
        "Met" if criteria["Accident ROC-AUC >= 0.75"] else "Missed",
    ),
    ("Confidence ranges", f"The 80% range contained the actual volume {metrics['coverage']['volume_80']:.1f}% of the time.", "78 to 82%", "Met" if criteria["80% band covers 78-82% of test values"] else "Missed"),
]
table = pd.DataFrame(rows, columns=["What it predicts", "Result", "Target", "Outcome"])
st.dataframe(table.style.map(lambda v: "color:#1b7a43;font-weight:600" if v == "Met" else "color:#c8102e;font-weight:600", subset=["Outcome"]), hide_index=True, width="stretch")
st.markdown(
    "**The main weak spots:** forecasts run too high in rain and fog, because a forecast can't know the weather in the period it's predicting. Accident risk works as a rough ranking only. "
    "Since this data looks synthetic, these figures don't show how it would do on real roads. Model performance has the details."
)
real_path = REPORTS / "real_data_results.json"
if real_path.exists():
    real = json.loads(real_path.read_text(encoding="utf-8"))
    st.markdown(
        f"**Does it work on real traffic?** For volume, yes. The same approach was also tested on real sensor counts from the I-94 highway in Minnesota, where it was off by {real['headline']['test_wape_pct']:.1f}% on average, one hour ahead. "
        "That data only has traffic counts, so congestion, travel time and accident risk remain untested on real roads."
    )
    st.page_link("views/real_data_check.py", label="See the real-data check", icon=":material/fact_check:")

st.subheader("How it works")
st.markdown(
    """
1. **Clean the data.** Sensor readings, weather and a calendar are checked, repaired and merged. Every change is logged.
2. **Describe each moment.** For each road, the system summarises recent traffic, the usual level for that time, weather, and the calendar. It only uses what would've been known at that moment.
3. **Train and compare models.** Thirteen forecasting approaches were compared for volume, from simple rules to neural networks, plus further models for the other forecasts. Past dates trained them, and later dates chose between them.
4. **Forecast, with confidence.** The chosen models produce the forecasts on these pages, with ranges based on how large their errors were.
"""
)

with st.expander("Where are the PRD's nine required views?"):
    st.markdown(
        """
| PRD view | Page |
|---|---|
| Live prediction | Live prediction (replay) |
| Historical trends | Historical trends |
| Congestion heatmap | Congestion heatmap |
| Road comparison | Road comparison |
| Model performance | Model performance |
| Feature importance | What drives the forecasts |
| Forecast visualisation | Forecasts vs what happened |
| Prediction confidence | Prediction confidence |
| Weather vs traffic | Weather vs traffic |

The PRD's landing surfaces are the first links above: the congestion heatmap for analysts and road comparison for planners.
"""
    )
with st.expander("Glossary"):
    for name, meaning in GLOSSARY.items():
        st.markdown(f"**{name}:** {meaning}")
