"""FlowCast dashboard (M7). Run with: streamlit run dashboard/app.py

It reads saved outputs and never trains on load. The retrain page only starts a run when its button is pressed.
"""

import streamlit as st
from common import has_upload

st.set_page_config(page_title="FlowCast", layout="wide")

pages = {
    "Start here": [
        st.Page("views/home.py", title="Overview", default=True),
    ],
    "What happens next?": [
        st.Page("views/live_prediction.py", title="Live prediction (replay)"),
        st.Page("views/accident_risk.py", title="Accident risk"),
        st.Page("views/prediction_interface.py", title="Request a forecast"),
    ],
    "What has been happening?": [
        st.Page("views/congestion_heatmap.py", title="Congestion heatmap"),
        st.Page("views/road_comparison.py", title="Road comparison"),
        st.Page("views/historical_trends.py", title="Historical trends"),
        st.Page("views/weather_vs_traffic.py", title="Weather vs traffic"),
    ],
    "Can I trust it?": [
        st.Page("views/forecast_visualisation.py", title="Forecasts vs what happened"),
        st.Page("views/prediction_confidence.py", title="Prediction confidence"),
        st.Page("views/model_performance.py", title="Model performance"),
        st.Page("views/real_data_check.py", title="Real-data check"),
        st.Page("views/feature_importance.py", title="What drives the forecasts"),
        st.Page("views/data_quality.py", title="Data quality"),
        st.Page("views/model_records.py", title="Model records"),
    ],
    "Tools": [
        st.Page("views/data_upload.py", title="Upload data"),
        st.Page("views/model_training.py", title="Retrain a model"),
        st.Page("views/reports.py", title="Reports and insights"),
    ],
}
page = st.navigation(pages, expanded=True)

st.sidebar.caption("Replay of a supplied 2025 dataset. Not connected to real roads.")
if has_upload():
    st.sidebar.warning("Forecasts include your uploaded data. Remove it on the Upload data page.", icon=":material/upload_file:")

page.run()
