import pandas as pd
import plotly.express as px
import streamlit as st
from common import (
    HORIZON_LABELS,
    LABELS,
    SEVERITY_COLOURS,
    model_details,
    moment_picker,
    page_header,
    predictions,
    segments,
    takeaway,
    terms,
)

page_header(
    "Live prediction (replay)",
    "If it were this moment, what happens next on every road?",
    "A replay of past dates, not a live feed. Pick a moment to treat as now, and each forecast uses only what was known at that moment. It starts on a weekday morning, when there's most to see.",
)

forecasts = predictions()
now, horizon = moment_picker(forecasts)
view = forecasts[(forecasts["issue_ts"] == now) & (forecasts["horizon"] == horizon)].merge(segments()[["road_id", "road_name", "latitude", "longitude"]], on="road_id")
if view.empty:
    st.warning("There's no forecast for this combination, because its target falls outside the period covered. Try a different time or a shorter look-ahead.")
    st.stop()

target = view["target_ts"].iat[0]
jammed = view[view["congestion_label"].isin(["Heavy", "Severe"])]
if jammed.empty:
    takeaway(f"At {target:%H:%M on %d %B}, no road is forecast to be in Heavy or Severe congestion.")
else:
    worst = jammed.loc[jammed["volume_pred"].idxmax(), "road_name"]
    takeaway(f"At {target:%H:%M on %d %B}, {len(jammed)} of {len(view)} roads are forecast to be in Heavy or Severe congestion. The busiest of those is {worst}.")

m1, m2, m3 = st.columns(3)
m1.metric("Time to drive the whole corridor", f"{view['travel_time_pred'].sum():.0f} min", help="The sum of all 25 forecast travel times.")
m2.metric("Vehicles across all roads", f"{view['volume_pred'].sum():,.0f}")
m3.metric("Heavy or Severe roads", f"{len(jammed)} of {len(view)}")

st.subheader("Where is the congestion?")
figure = px.scatter(
    view,
    x="longitude",
    y="latitude",
    color="congestion_label",
    size="volume_pred",
    hover_name="road_name",
    color_discrete_map=SEVERITY_COLOURS,
    category_orders={"congestion_label": LABELS},
    hover_data={"road_id": True, "volume_pred": ":.0f", "latitude": False, "longitude": False, "congestion_label": True},
    labels={"congestion_label": "Congestion", "volume_pred": "Vehicles", "road_id": "Road ID"},
)
figure.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Longitude (west to east)", yaxis_title="Latitude (south to north)")
figure.update_yaxes(scaleanchor="x", scaleratio=1)
st.plotly_chart(figure, width="stretch")
st.caption("Each dot is a road segment at its map position. A bigger dot means more vehicles.")

st.subheader("Every road")
table = pd.DataFrame(
    {
        "Road": view["road_name"],
        "Vehicles": view["volume_pred"].round().astype(int),
        "Likely range": view["volume_lo80"].round().astype(int).astype(str) + " to " + view["volume_hi80"].round().astype(int).astype(str),
        "Congestion": view["congestion_label"],
        "How sure": view["congestion_confidence"],
        "Speed (km/h)": view["speed_pred"].round(1),
        "Travel time (min)": view["travel_time_pred"].round(1),
    }
).sort_values("Vehicles", ascending=False)
styled = table.style.format({"How sure": "{:.0%}", "Speed (km/h)": "{:.1f}", "Travel time (min)": "{:.1f}"}).map(lambda v: f"background-color:{SEVERITY_COLOURS[v]};color:white", subset=["Congestion"])
st.dataframe(styled, hide_index=True, width="stretch", height=480)
st.caption(f"Forecast made for {HORIZON_LABELS[horizon]} ahead. The likely range should contain the actual number 8 times out of 10. For accident risk, see the Accident risk page.")
terms("Forecast window", "Volume", "Congestion level", "80% range")
model_details(view, ("volume_model", "speed_model", "travel_time_model", "congestion_model"))
