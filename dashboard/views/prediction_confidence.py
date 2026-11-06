import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import (
    HORIZON_LABELS,
    LABELS,
    SEVERITY_COLOURS,
    model_details,
    page_header,
    predictions,
    report_csv,
    segment_picker,
    takeaway,
    terms,
)

page_header(
    "Prediction confidence",
    "How much can each forecast be trusted?",
    "Every forecast comes with a range. The darker band should contain the actual value 8 times out of 10, and the lighter band 19 times out of 20. Wider bands mean more uncertainty, usually at peak.",
)

forecasts = predictions()
c1, c2, c3 = st.columns(3)
with c1:
    road = segment_picker()
horizon = c2.selectbox("Look ahead", list(HORIZON_LABELS), format_func=HORIZON_LABELS.get)
measure = c3.selectbox("Measure", ["volume", "speed", "travel_time"], format_func=lambda m: {"volume": "Vehicles", "speed": "Speed (km/h)", "travel_time": "Travel time (min)"}[m])

coverage = report_csv("metrics_by_horizon.csv")
row = coverage[coverage["horizon"] == horizon].iloc[0]
takeaway(f"The ranges hold up. On test dates at this look-ahead, the 80% range contained the actual value {row[f'{measure}_cov80']:.1f}% of the time, and the 95% range {row[f'{measure}_cov95']:.1f}%.")

rows = forecasts[(forecasts["road_id"] == road) & (forecasts["split"] == "test") & (forecasts["horizon"] == horizon)].sort_values("target_ts")
days = sorted(rows["target_ts"].dt.strftime("%Y-%m-%d").unique())
day_range = st.select_slider("Dates", options=days, value=(days[0], days[min(2, len(days) - 1)]))
rows = rows[rows["target_ts"].dt.strftime("%Y-%m-%d").between(*day_range)]

figure = go.Figure()
for level, shade in [(95, "rgba(42,111,151,0.15)"), (80, "rgba(42,111,151,0.3)")]:
    figure.add_scatter(x=rows["target_ts"], y=rows[f"{measure}_hi{level}"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    figure.add_scatter(x=rows["target_ts"], y=rows[f"{measure}_lo{level}"], fill="tonexty", fillcolor=shade, line=dict(width=0), name=f"{level}% range")
figure.add_scatter(x=rows["target_ts"], y=rows[f"{measure}_pred"], name="Forecast", line=dict(color="#2a6f97", width=1.6))
figure.add_scatter(x=rows["target_ts"], y=rows[f"actual_{measure}"], name="What happened", line=dict(color="#222", width=1.2))
figure.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.08), yaxis_title={"volume": "Vehicles per 30 minutes", "speed": "Speed (km/h)", "travel_time": "Travel time (min)"}[measure])
st.plotly_chart(figure, width="stretch")

with st.expander("See how the ranges held up at every look-ahead"):
    names = {"minutes_ahead": "Minutes ahead", f"{measure}_cov80": "80% range contained it", f"{measure}_cov95": "95% range contained it"}
    if measure == "volume":
        names.update({"volume_cov80_peak": "80% range, at peak", "volume_cov80_offpeak": "80% range, off-peak"})
    percent = {label: "{:.1f}%" for column, label in names.items() if column != "minutes_ahead"}
    st.dataframe(coverage[list(names)].rename(columns=names).style.format(percent), hide_index=True)

st.subheader("How sure is the congestion call?")
st.caption("Each congestion forecast comes with a probability. The right-hand chart checks whether those probabilities are honest: when it says 70% sure, is it right about 70% of the time?")
selected = predictions(include_uploads=False)
selected = selected[(selected["split"] == "test") & (selected["horizon"] == horizon)]
left, right = st.columns(2)
histogram = px.histogram(
    selected,
    x="congestion_confidence",
    color="congestion_label",
    nbins=30,
    color_discrete_map=SEVERITY_COLOURS,
    category_orders={"congestion_label": LABELS},
    labels={"congestion_confidence": "How sure the forecast was", "congestion_label": "Forecast level"},
)
histogram.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Forecasts", barmode="stack", xaxis_tickformat=".0%")
left.plotly_chart(histogram, width="stretch")

measured = selected[selected["target_valid"]].assign(
    band=lambda d: (d["congestion_confidence"] * 10).clip(upper=9.99).astype(int) / 10,
    hit=lambda d: d["congestion_pred"] == d["actual_congestion"],
)
reliable = measured.groupby("band").agg(claimed=("congestion_confidence", "mean"), observed=("hit", "mean"), n=("hit", "size")).reset_index()
reliability = go.Figure()
reliability.add_scatter(x=[0, 1], y=[0, 1], line=dict(color="#999", dash="dash"), name="Perfectly honest")
reliability.add_scatter(x=reliable["claimed"], y=reliable["observed"], mode="lines+markers", name="FlowCast", line=dict(color="#2a6f97"), customdata=reliable["n"], hovertemplate="Said %{x:.0%} sure, right %{y:.0%} of the time (%{customdata} forecasts)<extra></extra>")
reliability.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="How sure it said it was", yaxis_title="How often it was right", xaxis_tickformat=".0%", yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.1))
right.plotly_chart(reliability, width="stretch")
terms("80% range", "Peak", "Validation and test")
model_details(selected, ("volume_model", "congestion_model"))
