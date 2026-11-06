import plotly.express as px
import streamlit as st
from common import dataset, page_header, road_names, takeaway, terms

page_header(
    "Road comparison",
    "Which roads are busiest, slowest and least dependable?",
    "Choose roads to compare side by side, using measured periods from January to May. The ranking below covers all 25.",
)

names = road_names()
rows = dataset()
rows = rows[~rows["target_imputed"]]
options = list(names)
chosen = st.multiselect("Roads to compare", options, default=options[:5], format_func=names.get)
period = st.radio("Time of day", ["All day", "Peak only", "Off-peak only"], horizontal=True)
if not chosen:
    st.info("Choose at least one road to compare.")
    st.stop()

peak = rows["time"].between("07:30", "09:59") | rows["time"].between("17:00", "19:59")
rows = rows[peak] if period == "Peak only" else rows[~peak] if period == "Off-peak only" else rows

summary = (
    rows.groupby("road_id")
    .agg(
        volume=("traffic_volume", "mean"),
        speed=("avg_speed", "mean"),
        median_tt=("travel_time", "median"),
        p95_tt=("travel_time", lambda s: s.quantile(0.95)),
        severe=("congestion_level", lambda s: (s == "Severe").mean() * 100),
        accidents=("accident_count", lambda s: (s > 0).mean() * 100),
    )
    .reset_index()
)
summary["reliability"] = summary["p95_tt"] / summary["median_tt"]
summary["Road"] = summary["road_id"].map(names)
chart = summary[summary["road_id"].isin(chosen)]

takeaway(
    f"Across all 25 roads, {period.lower()}: {names[summary.loc[summary['volume'].idxmax(), 'road_id']]} is busiest, {names[summary.loc[summary['speed'].idxmin(), 'road_id']]} is slowest, "
    f"and {names[summary.loc[summary['reliability'].idxmax(), 'road_id']]} is least dependable, with a bad day taking {summary['reliability'].max():.1f} times as long as a typical one."
)

c1, c2, c3 = st.columns(3)
for column, field, label in [(c1, "volume", "Vehicles per 30 minutes"), (c2, "speed", "Average speed (km/h)"), (c3, "reliability", "Bad day vs typical day (travel time)")]:
    figure = px.bar(chart.sort_values(field), x=field, y="Road", orientation="h", labels={field: label, "Road": ""}, color_discrete_sequence=["#2a6f97"])
    figure.update_layout(height=max(280, 40 * len(chart)), margin=dict(l=10, r=10, t=10, b=10))
    column.plotly_chart(figure, width="stretch")

st.subheader("Traffic through the day")
profile = rows[rows["road_id"].isin(chosen)].groupby(["road_id", "time"])["traffic_volume"].mean().reset_index()
profile["Road"] = profile["road_id"].map(names)
figure = px.line(profile, x="time", y="traffic_volume", color="Road", labels={"time": "Time of day", "traffic_volume": "Average vehicles per 30 minutes"})
figure.update_layout(height=370, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(figure, width="stretch")

st.subheader("All 25 roads")
display = summary[["Road", "volume", "speed", "median_tt", "p95_tt", "reliability", "severe", "accidents"]].sort_values("volume", ascending=False)
display = display.rename(
    columns={
        "volume": "Vehicles per 30 min",
        "speed": "Average speed (km/h)",
        "median_tt": "Typical travel time (min)",
        "p95_tt": "Bad-day travel time (min)",
        "reliability": "Bad vs typical",
        "severe": "Time in Severe (%)",
        "accidents": "Periods with an accident (%)",
    }
)
st.dataframe(display.style.format(precision=2), hide_index=True, width="stretch")
terms("Reliability", "Peak", "Congestion level")
