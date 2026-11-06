import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import CONFIG, ROOT, dataset, page_header, road_names, segment_picker, takeaway, terms

page_header(
    "Historical trends",
    "What does normal look like on this road?",
    "Traffic and speed from January to May, and the shape of a typical day. Choose a road to see its pattern.",
)

rows = dataset()
c1, c2 = st.columns(2)
with c1:
    road = segment_picker()
grain = c2.radio("Show", ["One value per day", "Every 30 minutes"], horizontal=True)
days = sorted(rows["date"].unique())
daily_view = grain == "One value per day"
start, end = st.select_slider("Dates", options=days, value=(days[0], days[-1]) if daily_view else (days[-21], days[-14]))
part = rows[(rows["road_id"] == road) & rows["date"].between(start, end)]

measured = rows[(rows["road_id"] == road) & ~rows["target_imputed"]]
profile = measured.assign(period=lambda d: d["ts"].dt.day_of_week.ge(5).map({True: "Weekend", False: "Weekday"})).groupby(["period", "time"])[["traffic_volume", "avg_speed"]].mean().reset_index()
weekday = profile[profile["period"] == "Weekday"]
weekend = profile[profile["period"] == "Weekend"]
busiest = weekday.loc[weekday["traffic_volume"].idxmax()]
takeaway(
    f"{road_names()[road]} is busiest at {busiest['time']} on weekdays, with about {busiest['traffic_volume']:.0f} vehicles per 30 minutes at {busiest['avg_speed']:.0f} km/h. "
    f"Weekends peak at about {weekend['traffic_volume'].max():.0f} vehicles."
)

if daily_view:
    daily = part.groupby("date")[["traffic_volume", "avg_speed"]].mean().reset_index()
    x, volume, speed = pd.to_datetime(daily["date"]), daily["traffic_volume"], daily["avg_speed"]
else:
    x, volume, speed = part["ts"], part["traffic_volume"], part["avg_speed"]

st.subheader("Traffic over time")
overlay_path = ROOT / CONFIG["paths"]["processed"] / "bilstm_overlay.parquet"
figure = go.Figure()
figure.add_scatter(x=x, y=volume, name="Vehicles", line=dict(color="#2a6f97", width=1.3))
if not daily_view and overlay_path.exists():
    overlay = pd.read_parquet(overlay_path)
    overlay["target_ts"] = overlay["target_ts"].dt.tz_localize(None)
    overlay = overlay[(overlay["road_id"] == road) & overlay["target_ts"].between(part["ts"].min(), part["ts"].max())]
    if len(overlay):
        figure.add_scatter(x=overlay["target_ts"], y=overlay["bilstm"], name="Smoothed view (neural network)", line=dict(color="#ee7b30", width=1.3))
figure.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Vehicles per 30 minutes", legend=dict(orientation="h", y=1.1))
st.plotly_chart(figure, width="stretch")
if not daily_view:
    st.caption("The smoothed line is available from 16 April. It comes from a bidirectional LSTM, shown for context only and not used for forecasts.")
else:
    st.caption("Dips usually mark weekends, holidays or roadworks. Spikes usually mark local events.")

speed_figure = px.line(x=x, y=speed, labels={"x": "", "y": "Speed (km/h)"}, color_discrete_sequence=["#6a4c93"])
speed_figure.update_layout(height=270, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(speed_figure, width="stretch")

st.subheader("A typical day")
left, right = st.columns(2)
for column, field, label in [(left, "traffic_volume", "Average vehicles per 30 minutes"), (right, "avg_speed", "Average speed (km/h)")]:
    chart = px.line(profile, x="time", y=field, color="period", labels={"time": "Time of day", field: label, "period": ""})
    for start_time, end_time in CONFIG["peak_windows"]:
        chart.add_vrect(x0=start_time, x1=end_time, fillcolor="#ee7b30", opacity=0.08, line_width=0)
    chart.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    column.plotly_chart(chart, width="stretch")
st.caption("Shaded bands mark the peak periods.")
terms("Volume", "Peak")
