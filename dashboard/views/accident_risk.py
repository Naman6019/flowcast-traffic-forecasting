import pandas as pd
import plotly.express as px
import streamlit as st
from common import (
    HORIZON_LABELS,
    dataset,
    model_details,
    moment_picker,
    page_header,
    predictions,
    report_json,
    road_names,
    takeaway,
    terms,
)

page_header(
    "Accident risk",
    "Where should patrols be positioned right now?",
    "Roads ranked from most to least likely to have an accident. This is a rough guide, not a warning system.",
)

forecasts = predictions()
accident = report_json("final_metrics.json")["accident"]
st.warning(
    f"**Use this as a ranking, not as an alert.** Accidents are rare and hard to predict in this data. Between one period with an accident and one without, the model ranks the accident period higher {accident['roc_auc']:.0%} of the time, "
    f"where 50% would be a coin toss. The periods it ranks in its top 10% saw {accident['top_decile_lift']:.1f} times the usual accident rate on test dates.",
    icon=":material/warning:",
)

now, horizon = moment_picker(forecasts)
view = forecasts[(forecasts["issue_ts"] == now) & (forecasts["horizon"] == horizon)].copy()
if view.empty:
    st.warning("There's no forecast for this combination, because its target falls outside the period covered. Try a different time or a shorter look-ahead.")
    st.stop()

names = road_names()
# A road is "higher risk" if its score would be in the top 10% of all periods on validation dates.
reference = forecasts[(forecasts["split"] == "val") & (forecasts["horizon"] == horizon)]["accident_score"]
cutoff = reference.quantile(0.9)
view["higher"] = view["accident_score"] >= cutoff
view["times_usual"] = view["accident_prob"] / accident["base_rate"]
view = view.sort_values("accident_score", ascending=False)
higher = view[view["higher"]]
first = view.iloc[0]
target = first["target_ts"]
if higher.empty:
    takeaway(f"For {target:%H:%M on %d %B}, no road stands out. The highest-ranked is {names[first['road_id']]}, with a {first['accident_prob']:.1%} chance.")
else:
    busy = " Risk rises across the whole corridor at busy times, so at a moment like this, the order matters more than the group." if len(higher) > len(view) / 2 else ""
    takeaway(f"For {target:%H:%M on %d %B}, {len(higher)} of {len(view)} roads are in the higher-risk group. Start with {names[first['road_id']]}, at a {first['accident_prob']:.1%} chance, or {first['times_usual']:.1f} times the usual rate.{busy}")

left, right = st.columns([3, 2])
with left:
    st.subheader("Roads ranked by risk")
    ranked = pd.DataFrame(
        {
            "Rank": range(1, len(view) + 1),
            "Road": view["road_id"].map(names),
            "Chance of an accident": view["accident_prob"],
            "Compared with usual": view["times_usual"],
            "Risk group": view["higher"].map({True: "Higher", False: "Normal"}),
        }
    )
    styled = ranked.style.format({"Chance of an accident": "{:.2%}", "Compared with usual": "{:.1f}x"}).map(lambda v: "font-weight:600;color:#c8102e" if v == "Higher" else "", subset=["Risk group"])
    st.dataframe(styled, hide_index=True, width="stretch", height=560)
with right:
    st.subheader("Does the risk change soon?")
    roads = view["road_id"].head(6).tolist()
    trend = forecasts[(forecasts["issue_ts"] == now) & forecasts["road_id"].isin(roads)].assign(Road=lambda d: d["road_id"].map(names), Ahead=lambda d: d["horizon"].map(HORIZON_LABELS))
    chart = px.line(trend.sort_values("horizon"), x="Ahead", y="accident_prob", color="Road", markers=True, labels={"accident_prob": "Chance of an accident", "Ahead": "Look ahead"})
    chart.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_tickformat=".1%", legend=dict(orientation="h", y=-0.35))
    st.plotly_chart(chart, width="stretch")
    st.caption("The six highest-ranked roads, for each look-ahead.")

st.subheader("Where have accidents happened?")
rows = dataset()
rows = rows[~rows["target_imputed"]]
history = rows.groupby("road_id")["accident_count"].sum().sort_values().reset_index()
history["Road"] = history["road_id"].map(names)
chart = px.bar(history, x="accident_count", y="Road", orientation="h", labels={"accident_count": "Reported accidents, January to May", "Road": ""}, color_discrete_sequence=["#6a4c93"])
chart.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(chart, width="stretch")
st.caption("Accident rates differ little between roads, which is one reason this forecast is hard. Rain, fog, time of day and how full the road is matter more.")
terms("ROC-AUC", "Forecast window", "Validation and test")
model_details(view, ("accident_model",))
