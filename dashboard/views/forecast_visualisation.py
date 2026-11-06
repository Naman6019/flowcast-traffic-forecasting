import plotly.graph_objects as go
import streamlit as st
from common import HORIZON_LABELS, model_details, page_header, predictions, segment_picker, takeaway, terms

page_header(
    "Forecasts vs what happened",
    "How close are the forecasts to reality?",
    "The black line is what actually happened, and the coloured lines are what was forecast beforehand. The closer they are, the better.",
)

forecasts = predictions()
c1, c2, c3 = st.columns(3)
with c1:
    road = segment_picker()
split = c2.radio("Period", ["test", "val"], format_func={"test": "Test dates (9-31 May)", "val": "Validation dates (16 Apr-8 May)"}.get, horizontal=True)
chosen = c3.multiselect("Look ahead", list(HORIZON_LABELS), default=[1, 4], format_func=HORIZON_LABELS.get)

rows = forecasts[(forecasts["road_id"] == road) & (forecasts["split"] == split)]
days = sorted(rows["target_ts"].dt.strftime("%Y-%m-%d").unique())
start, end = st.select_slider("Dates", options=days, value=(days[0], days[min(6, len(days) - 1)]))
rows = rows[rows["target_ts"].dt.strftime("%Y-%m-%d").between(start, end)]

measured = rows[rows["target_valid"]].assign(
    error=lambda d: (d["volume_pred"] - d["actual_volume"]).abs(),
    squared=lambda d: (d["volume_pred"] - d["actual_volume"]) ** 2,
    pct=lambda d: (d["volume_pred"] - d["actual_volume"]).abs() / d["actual_volume"],
)
summary = measured.groupby("horizon").agg(RMSE=("squared", lambda s: s.mean() ** 0.5), MAE=("error", "mean"), MAPE=("pct", "mean")).reset_index()
if len(summary):
    near, far = summary.iloc[0], summary.iloc[-1]
    close = " Looking further ahead barely hurts accuracy here, which suggests most of the remaining error is unpredictable." if far["MAPE"] - near["MAPE"] < 0.01 else ""
    takeaway(f"On this road and these dates, forecasts 30 minutes ahead are off by {near['MAE']:.0f} vehicles on average ({near['MAPE']:.1%}), and forecasts 2 hours ahead by {far['MAE']:.0f} ({far['MAPE']:.1%}).{close}")

figure = go.Figure()
actual = rows[rows["horizon"] == rows["horizon"].min()].sort_values("target_ts")
figure.add_scatter(x=actual["target_ts"], y=actual["actual_volume"], name="What happened", line=dict(color="#222", width=2))
palette = ["#2a6f97", "#6a4c93", "#ee7b30", "#c8102e"]
for horizon in chosen:
    part = rows[rows["horizon"] == horizon].sort_values("target_ts")
    figure.add_scatter(x=part["target_ts"], y=part["volume_pred"], name="Forecast, " + HORIZON_LABELS[horizon] + " ahead", line=dict(color=palette[horizon - 1], width=1.3))
figure.update_layout(height=470, yaxis_title="Vehicles per 30 minutes", xaxis_title="", legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(figure, width="stretch")

st.subheader("Accuracy for this road and these dates")
display = summary.assign(horizon=summary["horizon"].map(HORIZON_LABELS)).rename(columns={"horizon": "Look ahead", "MAE": "Average error (vehicles)", "MAPE": "Average error (%)", "RMSE": "RMSE (vehicles)"})
st.dataframe(display[["Look ahead", "Average error (vehicles)", "Average error (%)", "RMSE (vehicles)"]].style.format({"RMSE (vehicles)": "{:.1f}", "Average error (vehicles)": "{:.1f}", "Average error (%)": "{:.1%}"}), hide_index=True)
terms("RMSE", "Validation and test")
model_details(rows)
