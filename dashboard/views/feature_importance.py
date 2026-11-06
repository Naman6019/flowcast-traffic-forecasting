import plotly.express as px
import streamlit as st
from common import FEATURE_NAMES, GROUP_NAMES, TARGET_NAMES, page_header, report_csv, takeaway

page_header(
    "What drives the forecasts",
    "What information do the predictions rely on?",
    "To measure how much a model relies on a kind of information, that information is scrambled and the model is tested again. A big drop in accuracy means the model depends on it.",
)

groups = report_csv("feature_importance_groups.csv")
groups = groups.assign(group=groups["group"].map(GROUP_NAMES))
volume = groups[groups["target"] == "volume"].sort_values("skill_change", ascending=False)
accident = groups[groups["target"] == "accident"].sort_values("skill_change", ascending=False)
takeaway(
    f"Volume forecasts mostly depend on two things: {volume.iloc[0]['group'].lower()} and {volume.iloc[1]['group'].lower()}. "
    f"Accident risk leans most on {accident.iloc[0]['group'].lower()} and {accident.iloc[1]['group'].lower()}. Weather adds little to volume forecasts, because the weather in the period being predicted isn't known yet."
)

left, right = st.columns(2)
for column, part, title, label in [(left, volume, "Traffic volume", "Extra error when scrambled (vehicles)"), (right, accident, "Accident risk", "Ranking skill lost when scrambled (ROC-AUC)")]:
    chart = px.bar(part.sort_values("skill_change"), x="skill_change", y="group", orientation="h", title=title, labels={"skill_change": label, "group": ""}, color_discrete_sequence=["#6a4c93"])
    chart.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    column.plotly_chart(chart, width="stretch")
st.caption("Measured on test dates, using the best non-neural model for each prediction. Near zero means the model doesn't rely on it.")

st.subheader("Individual inputs")
st.caption("A finer view from the tree-based models. Many inputs carry similar information and share the credit, so don't read too much into the exact order.")
table = report_csv("feature_importance.csv")
c1, c2, c3 = st.columns(3)
target = c1.selectbox("Prediction", sorted(table["target"].unique()), format_func=TARGET_NAMES.get)
model = c2.selectbox("Model", sorted(table.loc[table["target"] == target, "model"].unique()), format_func={"forest": "Random forest", "xgboost": "XGBoost"}.get)
count = c3.slider("Inputs shown", 5, 30, 12)
top = table[(table["target"] == target) & (table["model"] == model)].nlargest(count, "importance").sort_values("importance")
top = top.assign(Input=top["feature"].map(lambda f: FEATURE_NAMES.get(f, f)))
figure = px.bar(top, x="importance", y="Input", orientation="h", labels={"importance": "Share of the model's decisions", "Input": ""}, color_discrete_sequence=["#2a6f97"])
figure.update_layout(height=max(320, 28 * count), margin=dict(l=10, r=10, t=10, b=10), xaxis_tickformat=".0%")
st.plotly_chart(figure, width="stretch")
