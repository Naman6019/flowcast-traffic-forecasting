import pandas as pd
import plotly.express as px
import streamlit as st
from common import (
    GROUP_NAMES,
    HORIZON_LABELS,
    SEVERITY_COLOURS,
    model_details,
    page_header,
    predictions,
    registry,
    segment_picker,
    terms,
)

from src.features import load_features
from src.serve import explain_volume

page_header(
    "Request a forecast",
    "What will one road look like at one moment?",
    "Choose a road, a past moment to treat as now, and how far ahead to look. Because this is a replay, you can also see what actually happened.",
)

forecasts = predictions()
c1, c2, c3, c4 = st.columns(4)
with c1:
    road = segment_picker()
days = sorted(forecasts["issue_ts"].dt.strftime("%Y-%m-%d").unique())
day = c2.selectbox("Pretend today is", days, index=len(days) - 2)
times = sorted(forecasts.loc[forecasts["issue_ts"].dt.strftime("%Y-%m-%d") == day, "issue_ts"].dt.strftime("%H:%M").unique())
clock = c3.selectbox("And the time is just after", times, index=times.index("08:00") if "08:00" in times else 0)
horizon = c4.selectbox("Look ahead", list(HORIZON_LABELS), format_func=HORIZON_LABELS.get)

selected = forecasts[(forecasts["road_id"] == road) & (forecasts["issue_ts"] == pd.Timestamp(f"{day} {clock}")) & (forecasts["horizon"] == horizon)]
if selected.empty:
    st.warning("There's no forecast for this combination, because its target falls outside the period covered. Try a different time or a shorter look-ahead.")
    st.stop()
row = selected.iloc[0]

st.subheader(f"Forecast for {row['target_ts']:%H:%M on %d %B %Y}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Vehicles", f"{row['volume_pred']:.0f}", f"Actual: {row['actual_volume']:.0f}", delta_color="off", delta_arrow="off")
m1.caption(f"Likely range: {row['volume_lo80']:.0f} to {row['volume_hi80']:.0f}")
m2.metric("Congestion", row["congestion_label"], f"Actual: {row['actual_label']}", delta_color="off", delta_arrow="off")
m2.caption(f"{row['congestion_confidence']:.0%} sure")
m3.metric("Travel time", f"{row['travel_time_pred']:.1f} min", f"Actual: {row['actual_travel_time']:.1f} min", delta_color="off", delta_arrow="off")
m3.caption(f"Likely range: {row['travel_time_lo80']:.1f} to {row['travel_time_hi80']:.1f} min")
m4.metric("Chance of an accident", f"{row['accident_prob']:.2%}", "Accident reported" if row["actual_accident"] == 1 else "None reported", delta_color="off", delta_arrow="off")
m4.caption("Treat this as a rough guide.")
if not row["target_valid"]:
    st.info("The sensor log has no reading for this period, so the actual values shown are estimates.")

left, right = st.columns(2)
with left:
    st.subheader("How likely is each congestion level?")
    classes = pd.DataFrame({"Level": ["Free-flow", "Moderate", "Heavy", "Severe"], "Probability": [row["p_free_flow"], row["p_moderate"], row["p_heavy"], row["p_severe"]]})
    chart = px.bar(classes, x="Probability", y="Level", orientation="h", color="Level", color_discrete_map=SEVERITY_COLOURS, category_orders={"Level": ["Severe", "Heavy", "Moderate", "Free-flow"]}, labels={"Level": ""})
    chart.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, xaxis_tickformat=".0%")
    st.plotly_chart(chart, width="stretch")
with right:
    st.subheader("What pushed the vehicle count up or down?")
    frame = load_features(horizon)
    frame = frame[(frame["road_id"] == road) & (frame["ts"].dt.tz_localize(None) == row["issue_ts"])]
    contributions = explain_volume(frame, horizon, registry())
    if contributions is None:
        st.info("No explanation is available for this forecast.")
    else:
        contributions = contributions.assign(group=contributions["group"].map(GROUP_NAMES))
        chart = px.bar(contributions.sort_values("contribution"), x="contribution", y="group", orientation="h", labels={"contribution": "Vehicles added or removed", "group": ""}, color_discrete_sequence=["#2a6f97"])
        chart.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(chart, width="stretch")
        st.caption("How the forecast changes if that information is swapped for its typical value. This comes from the random forest, which is easier to probe than the neural network and forecasts similar values.")

terms("Forecast window", "Congestion level", "80% range")
model_details(selected, ("volume_model", "speed_model", "travel_time_model", "congestion_model", "accident_model"))
st.caption(f"Input fingerprint: `{row['input_hash']}`. It identifies exactly what data this forecast used.")
