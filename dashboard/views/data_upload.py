import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from common import HORIZON_LABELS, LABELS, has_upload, page_header, road_names

from src.serve import UploadError, clear_uploaded, run_upload, save_uploaded

page_header(
    "Upload data",
    "What would the forecasts be with new sensor data?",
    "Upload a file in the traffic sensor log format. It goes through the same checks and cleaning as the main data, then the saved models forecast from it. Nothing retrains.",
)

if has_upload():
    st.success("Your last upload is showing on the forecast pages.", icon=":material/check_circle:")
    if st.button("Remove uploaded data, and go back to the original forecasts"):
        clear_uploaded()
        st.session_state.pop("upload_result", None)
        st.rerun()

uploaded = st.file_uploader("Traffic sensor log (CSV)", type="csv")
if uploaded is None:
    st.info("Dates must fall between 1 January and 31 May 2025, because that's what the weather and calendar data covers. Earlier rows come from the stored data, so even a single day can be forecast.")
    st.stop()

if st.button("Check the file and forecast", type="primary"):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "upload.csv"
        path.write_bytes(uploaded.getvalue())
        try:
            with st.spinner("Checking, cleaning and forecasting"):
                result = run_upload(path)
        except UploadError as error:
            st.error(str(error))
            st.stop()
    save_uploaded(result["predictions"])
    st.session_state["upload_result"] = result
    st.rerun()

result = st.session_state.get("upload_result")
if result is None:
    st.stop()

forecasts = result["predictions"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows received", f"{result['rows_received']:,}")
m2.metric("Forecasts made", f"{len(forecasts):,}")
m3.metric("Problems logged", f"{len(result['quarantine']):,}")
m4.metric("Time taken", f"{result['seconds']:.1f} s")

start, end = forecasts["issue_ts"].min(), forecasts["issue_ts"].max()
st.subheader("See the results on the other pages")
st.markdown(f"Forecasts for {start:%d %B %H:%M} to {end:%d %B %H:%M} now use your data. Accuracy figures and the Model performance page don't change, because they describe the evaluated models.")
c1, c2, c3 = st.columns(3)
with c1:
    st.page_link("views/live_prediction.py", label="Live prediction", icon=":material/schedule:")
with c2:
    st.page_link("views/congestion_heatmap.py", label="Congestion heatmap", icon=":material/grid_on:")
with c3:
    st.page_link("views/accident_risk.py", label="Accident risk", icon=":material/emergency:")

st.subheader("What was wrong with the file")
quarantine = result["quarantine"]
if quarantine.empty:
    st.success("No rejected rows or repaired values.")
else:
    plain = {
        "negative_volume": "Negative vehicle count",
        "speed_out_of_range": "Impossible speed",
        "occupancy_out_of_range": "Occupancy above 100%",
        "invalid_timestamp": "Date or time couldn't be read",
        "unknown_segment": "Road not in this corridor",
        "duplicate_key": "Same reading logged twice",
        "row_added_to_grid": "No reading for a period",
        "volume_from_vehicle_count": "Vehicle count rebuilt",
        "speed_from_travel_time": "Speed rebuilt",
        "occupancy_blanked": "Occupancy removed",
        "congestion_filled": "Blank congestion level filled",
        "interpolated": "Estimated from neighbouring readings",
        "window_median": "Estimated from the usual level",
        "invalid_json": "Vehicle mix couldn't be read",
        "unparseable_number": "Number couldn't be read",
        "travel_time_from_speed": "Travel time worked out from speed",
        "accident_count_zero": "Blank accident count set to zero",
        "off_grid_time": "Time not on a 30-minute boundary",
        "invalid_capacity": "Invalid road capacity",
        "missing_road_id": "No road given",
    }
    summary = quarantine.groupby("rule").size().rename("Entries").sort_values(ascending=False).reset_index()
    summary["Problem or fix"] = summary["rule"].map(lambda r: plain.get(r, r.replace("_", " ").capitalize()))
    st.dataframe(summary[["Problem or fix", "Entries"]], hide_index=True, width="stretch")
    st.download_button("Download the full log", quarantine.to_csv(index=False), "quarantine.csv", "text/csv")

st.subheader("The forecasts")
names = road_names()
display = pd.DataFrame(
    {
        "Road": forecasts["road_id"].map(names),
        "Forecast made": forecasts["issue_ts"].dt.tz_localize(None),
        "For": forecasts["target_ts"].dt.tz_localize(None),
        "Look ahead": forecasts["horizon"].map(HORIZON_LABELS),
        "Vehicles": forecasts["volume_pred"],
        "Likely low": forecasts["volume_lo80"],
        "Likely high": forecasts["volume_hi80"],
        "Congestion": forecasts["congestion_pred"].map(dict(enumerate(LABELS))),
        "Travel time (min)": forecasts["travel_time_pred"],
        "Chance of an accident": forecasts["accident_prob"],
    }
)
st.dataframe(display.style.format({"Vehicles": "{:.0f}", "Likely low": "{:.0f}", "Likely high": "{:.0f}", "Travel time (min)": "{:.1f}", "Chance of an accident": "{:.2%}"}), hide_index=True, width="stretch", height=420)
st.download_button("Download the forecasts", forecasts.to_csv(index=False), "flowcast_forecasts.csv", "text/csv")
