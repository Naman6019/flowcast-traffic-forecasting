import numpy as np
import plotly.express as px
import streamlit as st
from common import CONFIG, dataset, page_header, predictions, takeaway, terms

page_header(
    "Weather vs traffic",
    "What do rain and fog do to traffic, and to the forecasts?",
    "Weather here is what was observed at the time. A real forecast couldn't have known it in advance. Peak and off-peak are shown separately, so time of day doesn't distort the comparison.",
)

rows = dataset()
rows = rows[~rows["target_imputed"]].copy()
rows["period"] = np.where(rows["time"].between("07:30", "09:59") | rows["time"].between("17:00", "19:59"), "Peak", "Off-peak")
order = CONFIG["weather"]["vocabulary"]

summary = (
    rows.groupby(["period", "weather_condition"])
    .agg(Windows=("traffic_volume", "size"), Volume=("traffic_volume", "mean"), Speed=("avg_speed", "mean"), Accidents=("accident_count", lambda s: (s > 0).mean() * 100))
    .reset_index()
)
peak = summary[summary["period"] == "Peak"].set_index("weather_condition")
overall_rate = rows.assign(hit=rows["accident_count"] > 0).groupby("weather_condition")["hit"].mean() * 100
takeaway(
    f"At peak, rain cuts traffic by {1 - peak.loc['Rain', 'Volume'] / peak.loc['Clear', 'Volume']:.0%} and fog by {1 - peak.loc['Fog', 'Volume'] / peak.loc['Clear', 'Volume']:.0%} compared with clear weather. "
    f"Accidents are far more common: {overall_rate['Rain']:.1f}% of periods in rain and {overall_rate['Fog']:.1f}% in fog, against {overall_rate['Clear']:.1f}% in clear weather."
)

c1, c2, c3 = st.columns(3)
for column, field, label in [(c1, "Volume", "Vehicles per 30 minutes"), (c2, "Speed", "Average speed (km/h)"), (c3, "Accidents", "Periods with an accident (%)")]:
    figure = px.bar(
        summary,
        x="weather_condition",
        y=field,
        color="period",
        barmode="group",
        category_orders={"weather_condition": order},
        labels={"weather_condition": "", field: label, "period": ""},
        color_discrete_map={"Peak": "#ee7b30", "Off-peak": "#2a6f97"},
    )
    figure.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
    column.plotly_chart(figure, width="stretch")
st.caption(f"Treat fog at peak with caution: it rests on only {int(peak.loc['Fog', 'Windows'])} periods.")
with st.expander("See the numbers"):
    st.dataframe(
        summary.rename(columns={"period": "Time of day", "weather_condition": "Weather", "Windows": "Periods", "Volume": "Vehicles", "Speed": "Speed (km/h)", "Accidents": "With an accident (%)"}).style.format(
            {"Vehicles": "{:.0f}", "Speed (km/h)": "{:.1f}", "With an accident (%)": "{:.2f}", "Periods": "{:,}"}
        ),
        hide_index=True,
    )

st.subheader("Speed as rain and visibility change")
bucket = rows.assign(
    Rainfall=lambda d: np.select([d["rainfall"] == 0, d["rainfall"] < 2, d["rainfall"] < 5], ["Dry", "Under 2 mm", "2 to 5 mm"], "5 mm or more"),
    Visibility=lambda d: np.select([d["visibility"] < 1000, d["visibility"] < 3000, d["visibility"] < 8000], ["Under 1 km", "1 to 3 km", "3 to 8 km"], "8 km or more"),
)
left, right = st.columns(2)
for column, field, categories in [(left, "Rainfall", ["Dry", "Under 2 mm", "2 to 5 mm", "5 mm or more"]), (right, "Visibility", ["8 km or more", "3 to 8 km", "1 to 3 km", "Under 1 km"])]:
    grouped = bucket.groupby(["period", field])["avg_speed"].mean().reset_index()
    chart = px.line(grouped, x=field, y="avg_speed", color="period", markers=True, category_orders={field: categories}, labels={"avg_speed": "Average speed (km/h)", "period": ""})
    chart.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    column.plotly_chart(chart, width="stretch")

st.subheader("Do the forecasts cope with bad weather?")
forecasts = predictions(include_uploads=False)
test = forecasts[(forecasts["split"] == "test") & (forecasts["horizon"] == 1) & forecasts["target_valid"]]
joined = test.merge(rows[["road_id", "ts", "weather_condition"]], left_on=["road_id", "target_ts"], right_on=["road_id", "ts"])
accuracy = (
    joined.assign(error=lambda d: d["volume_pred"] - d["actual_volume"], pct=lambda d: (d["volume_pred"] - d["actual_volume"]).abs() / d["actual_volume"])
    .groupby("weather_condition")
    .agg(Forecasts=("error", "size"), Off=("pct", "mean"), Bias=("error", "mean"))
    .reindex(order)
)
takeaway(
    f"Not fully. Forecasts run {accuracy.loc['Rain', 'Bias']:.0f} vehicles too high in rain and {accuracy.loc['Fog', 'Bias']:.0f} too high in fog. The model can't know the weather in the period it's predicting, so it expects normal traffic. "
    "A live weather forecast would be the most useful addition."
)
st.dataframe(
    accuracy.reset_index().rename(columns={"weather_condition": "Weather", "Off": "Average error", "Bias": "Forecast too high (+) or too low (-)"}).style.format({"Forecasts": "{:,.0f}", "Average error": "{:.1%}", "Forecast too high (+) or too low (-)": "{:+.1f}"}),
    hide_index=True,
)
st.caption("Test dates (9 to 31 May), 30 minutes ahead.")
terms("Peak", "Validation and test")
