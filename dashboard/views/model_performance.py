import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import (
    HORIZON_LABELS,
    MODEL_NAMES,
    REPORTS,
    TARGET_NAMES,
    page_header,
    registry,
    report_csv,
    report_json,
    takeaway,
    terms,
)

page_header(
    "Model performance",
    "How accurate is the system, and where does it fall short?",
    "Scores are on test dates (9 to 31 May), which played no part in building or choosing the models. Only periods with a real measurement count.",
)

metrics = report_json("final_metrics.json")
persistence = report_csv("scoreboard_volume.csv").set_index("model").loc["Persistence (last window)"]
matrix = pd.read_csv(REPORTS / "confusion_congestion.csv", index_col=0)
values = matrix.to_numpy()
errors = values.sum() - values.trace()
adjacent = sum(values[i, i + 1] + values[i + 1, i] for i in range(3))

st.subheader("In plain terms")
plain = pd.DataFrame(
    [
        ("Traffic volume", f"Off by {metrics['volume']['mae']:.0f} vehicles on average ({metrics['volume']['mape_pct']:.1f}%). Repeating the latest reading would be off by {persistence['test_mae']:.0f} ({persistence['test_mape_pct']:.1f}%)."),
        ("Speed", f"Off by {metrics['speed']['mae']:.1f} km/h on average ({metrics['speed']['mape_pct']:.1f}%)."),
        ("Travel time", f"Off by {metrics['travel_time']['mae'] * 60:.0f} seconds on average ({metrics['travel_time']['mape_pct']:.1f}%)."),
        ("Congestion level", f"Exactly right {metrics['congestion']['accuracy']:.0%} of the time. {adjacent / errors:.0%} of mistakes are just one level off. Heavy and Severe are the hardest."),
        ("Accident risk", f"A weak signal: it ranks an accident period above a quiet one {metrics['accident']['roc_auc']:.0%} of the time, where a coin toss would be 50%. Useful as a rough ranking only."),
    ],
    columns=["What it predicts", "How well it does, 30 minutes ahead"],
)
st.dataframe(plain, hide_index=True, width="stretch")

st.subheader("Against the targets in the brief")
intervals = metrics["intervals_95"]
readable = {
    "Volume MAPE <= 12%": ("Volume error of at most 12%", "{:.2f}%", intervals["volume_mape_pct"]),
    "Congestion macro-F1 >= 0.80": ("Congestion macro-F1 of at least 0.80", "{:.3f}", intervals["congestion_macro_f1"]),
    "Accident ROC-AUC >= 0.75": ("Accident ROC-AUC of at least 0.75", "{:.3f}", intervals["accident_roc_auc"]),
    "Sequence model beats best classical on test RMSE": ("Neural network beats the best other model (lower RMSE wins)", "{:.2f}", None),
    "80% band covers 78-82% of test values": ("80% range contains 78 to 82% of actual values", "{:.1f}%", None),
    "Batch inference for one horizon <= 30 s": ("Full forecast run takes at most 30 seconds", "{:.1f} s", None),
}
rows = []
for item in metrics["criteria"]:
    label, fmt, interval = readable[item["criterion"]]
    rows.append({"Target": label, "Achieved": fmt.format(item["value"]), "Likely range": f"{fmt.format(interval[0])} to {fmt.format(interval[1])}" if interval else "", "Result": "Met" if item["met"] else "Not met"})
st.dataframe(pd.DataFrame(rows).style.map(lambda v: "color:#1b7a43;font-weight:600" if v == "Met" else "color:#c8102e;font-weight:600", subset=["Result"]), hide_index=True, width="stretch")
takeaway("Congestion and accident risk miss their targets, and the evidence points to the data rather than a lack of tuning: every type of model landed in a narrow range on both. Since this data looks synthetic, these scores don't show how the system would do on real roads. The Real-data check page tests volume on real traffic.")

st.subheader("Was the neural network worth it?")
benchmark = metrics["benchmark"]
gain = benchmark["selected_classical_test_rmse"] - benchmark["deep_test_rmse"]
st.markdown(
    f"**Not clearly.** The {MODEL_NAMES[benchmark['deep']]} beat every other model, which meets the brief's benchmark. But its typical error was only {gain:.2f} vehicles smaller than the {MODEL_NAMES[benchmark['selected_classical']].lower()}'s ({gain / benchmark['selected_classical_test_rmse']:.1%}), "
    "and it takes several times longer to run and is harder to interpret. Simpler models come very close."
)
volume = report_csv("scoreboard_volume.csv").dropna(subset=["test_rmse"]).sort_values("test_rmse", ascending=False)
volume["Model"] = volume["model"].map(lambda m: MODEL_NAMES.get(m, m))
chart = px.bar(volume, x="test_rmse", y="Model", orientation="h", labels={"test_rmse": "Typical forecast error, RMSE (vehicles, lower is better)", "Model": ""}, color_discrete_sequence=["#2a6f97"], text=volume["test_rmse"].round(1))
chart.update_layout(height=470, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(chart, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Where congestion calls go wrong")
    shares = matrix.div(matrix.sum(axis=1), axis=0)
    figure = px.imshow(shares, text_auto=".0%", color_continuous_scale="Blues", labels=dict(x="Forecast level", y="What actually happened", color="Share"), zmin=0, zmax=1)
    figure.update_layout(height=390, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(figure, width="stretch")
    st.caption("Each row adds up to 100%. The diagonal shows correct calls. Nearly all mistakes sit right next to it.")
with right:
    st.subheader("How well accidents are ranked")
    roc = report_csv("roc_accident.csv")
    figure = go.Figure()
    figure.add_scatter(x=[0, 1], y=[0, 1], line=dict(color="#999", dash="dash"), name="Coin toss")
    figure.add_scatter(x=roc["fpr"], y=roc["tpr"], line=dict(color="#c8102e"), name=f"FlowCast (ROC-AUC {metrics['accident']['roc_auc']:.3f})")
    figure.update_layout(height=390, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Quiet periods wrongly flagged", yaxis_title="Accident periods caught", xaxis_tickformat=".0%", yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(figure, width="stretch")
    st.caption("A curve hugging the dashed line is no better than chance. Further above it is better.")

st.subheader("Where volume errors are largest")
errors_table = report_csv("error_breakdown_volume.csv")
for tab, group in zip(st.tabs(["Hour of day", "Day of week", "Weather"]), ["hour", "day_of_week", "weather"], strict=True):
    part = errors_table[errors_table["group"] == group]
    if group == "hour":
        part = part.assign(value=part["value"].astype(int)).sort_values("value")
    if group == "day_of_week":
        part = part.set_index("value").reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]).reset_index()
    chart = px.bar(part, x="value", y="rmse", labels={"value": "", "rmse": "Typical error, RMSE (vehicles)", "n": "Forecasts", "mape_pct": "Average error (%)", "bias": "Too high (+) or low (-)"}, hover_data=["n", "mape_pct", "bias"], color_discrete_sequence=["#2a6f97"])
    tab.plotly_chart(chart.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10)), width="stretch")
st.caption("Errors are largest at the busiest times, in proportion to traffic. Thursday's test dates include a large music festival, and forecasts run too high in rain and fog.")

st.subheader("Technical details")
with st.expander("Scores for every model"):
    target = st.radio("Prediction", list(TARGET_NAMES), horizontal=True, format_func=TARGET_NAMES.get)
    board = report_csv(f"scoreboard_{target}.csv")
    primary = {"congestion": "macro_f1", "accident": "roc_auc"}.get(target, "rmse")
    extras = {"congestion": ["test_accuracy", "test_f1_class2", "test_f1_class3"], "accident": ["test_pr_auc", "test_top_decile_lift", "test_precision", "test_recall", "test_flag_rate"]}.get(target, ["test_mae", "test_mape_pct", "test_r2"])
    columns = ["model", "cv_mean", "cv_std", f"val_{primary}", f"test_{primary}", *extras, "infer_ms_per_1k", "interpretability"]
    display = board[[c for c in columns if c in board]].assign(model=board["model"].map(lambda m: MODEL_NAMES.get(m, m)))
    labels = {
        "model": "Model",
        "cv_mean": "Cross-validation score",
        "cv_std": "Spread across folds",
        f"val_{primary}": f"Validation {primary}",
        f"test_{primary}": f"Test {primary}",
        "test_mae": "Test MAE",
        "test_mape_pct": "Test MAPE (%)",
        "test_r2": "Test R²",
        "test_accuracy": "Test accuracy",
        "test_f1_class2": "Test F1, Heavy",
        "test_f1_class3": "Test F1, Severe",
        "test_pr_auc": "Test PR-AUC",
        "test_top_decile_lift": "Top 10% lift",
        "test_precision": "Test precision",
        "test_recall": "Test recall",
        "test_flag_rate": "Share of periods flagged",
        "infer_ms_per_1k": "ms per 1,000 forecasts",
        "interpretability": "How easy to interpret",
    }
    st.dataframe(display.rename(columns=labels).style.format(precision=3, na_rep="-"), hide_index=True, width="stretch")
    production = registry()["production"]
    st.caption(f"In service: {production['volume']} model for volume, and {MODEL_NAMES[production['congestion']['model']].lower()} for congestion. The reasoning is in reports/model_selection.md.")
with st.expander("Accuracy at each look-ahead"):
    horizon = report_csv("metrics_by_horizon.csv")
    horizon["Look ahead"] = horizon["horizon"].map(HORIZON_LABELS)
    names = {"volume_rmse": "Volume RMSE", "volume_mape_pct": "Volume MAPE (%)", "speed_rmse": "Speed RMSE", "travel_time_rmse": "Travel time RMSE", "congestion_macro_f1": "Congestion macro-F1", "accident_roc_auc": "Accident ROC-AUC"}
    st.dataframe(horizon[["Look ahead", *names]].rename(columns=names).style.format(precision=3), hide_index=True, width="stretch")
with st.expander("Full classical vs neural network comparison"):
    st.markdown((REPORTS / "benchmark.md").read_text(encoding="utf-8").split("\n", 2)[2])
for name, caption in [("final_volume_residuals.png", "Forecast against actual, and the errors, on test dates."), ("deep_loss_curves.png", "Training and validation loss for every neural network run.")]:
    path = REPORTS / "figures" / name
    if path.exists():
        with st.expander(caption):
            st.image(str(path))
terms("RMSE", "MAE", "MAPE", "R²", "Macro-F1", "ROC-AUC", "Validation and test")
