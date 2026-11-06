import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    LABELS,
    SEVERITY_SEQUENCE,
    dataset,
    model_details,
    page_header,
    predictions,
    road_names,
    takeaway,
    terms,
)

page_header(
    "Congestion heatmap",
    "Which roads were jammed, and when?",
    "Each row is a road and each column a 30-minute period of the day, from midnight on the left. Colour shows how congested it was. Compare actual against forecast to see what the system got right.",
)

forecasts = predictions()
h1 = forecasts[forecasts["horizon"] == 1]
days = sorted(h1["target_ts"].dt.strftime("%Y-%m-%d").unique())
col1, col2 = st.columns([1, 2])
day = col1.selectbox("Date", days, index=len(days) - 2)
mode = col2.radio("Show", ["What actually happened", "What was forecast, 30 minutes before"], horizontal=True)
is_actual = mode.startswith("What actually")

if is_actual:
    rows = dataset()
    rows = rows[rows["date"] == day].assign(window=lambda d: d["time"], code=lambda d: d["congestion_level"].map({v: i for i, v in enumerate(LABELS)}))
else:
    rows = h1[h1["target_ts"].dt.strftime("%Y-%m-%d") == day].assign(window=lambda d: d["target_ts"].dt.strftime("%H:%M"), code=lambda d: d["congestion_pred"])

names = road_names()
day_label = pd.Timestamp(day).strftime("%A %d %B")
counts = rows[rows["code"] >= 2].groupby("window")["road_id"].nunique()
if counts.empty:
    takeaway(f"No road reached Heavy or Severe congestion on {day_label}.")
else:
    road_totals = rows[rows["code"] >= 2].groupby("road_id")["window"].nunique()
    takeaway(f"On {day_label}, the worst time was {counts.idxmax()}, when {counts.max()} of {rows['road_id'].nunique()} roads were Heavy or Severe. {names[road_totals.idxmax()]} spent the longest congested, at {road_totals.max() / 2:.1f} hours.")

grid = rows.pivot_table(index="road_id", columns="window", values="code", aggfunc="first").sort_index(ascending=False)
scale = []
for i, colour in enumerate(SEVERITY_SEQUENCE):
    scale += [[i / 4, colour], [(i + 1) / 4, colour]]
figure = go.Figure(
    go.Heatmap(
        z=(grid + 0.5).to_numpy(),
        x=grid.columns,
        y=[names[r] for r in grid.index],
        zmin=0,
        zmax=4,
        colorscale=scale,
        customdata=grid.map(lambda c: LABELS[int(c)] if c == c else "No forecast").to_numpy(),
        hovertemplate="%{y} at %{x}: %{customdata}<extra></extra>",
        colorbar=dict(tickvals=[0.5, 1.5, 2.5, 3.5], ticktext=LABELS, title="Congestion"),
    )
)
figure.update_layout(height=640, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Time of day", yaxis_title="")
st.plotly_chart(figure, width="stretch")

st.caption("Share of all 30-minute periods on this date")
share = rows["code"].value_counts(normalize=True).reindex(range(4), fill_value=0)
for column, label, value in zip(st.columns(4), LABELS, share, strict=True):
    column.metric(label, f"{value:.0%}")
terms("Congestion level", "Forecast window")
if not is_actual:
    model_details(rows, ("volume_model", "congestion_model"))
