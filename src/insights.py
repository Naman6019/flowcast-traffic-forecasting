"""Reports and insights: a summary of any date range, built only from real data and saved forecasts (FR-18)."""

import markdown
import pandas as pd

from src.evaluate import accident_metrics, congestion_metrics, regression_metrics


def _table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def build_summary(data: pd.DataFrame, predictions: pd.DataFrame, start: str, end: str) -> str:
    """Markdown summary of a date range. Forecast accuracy appears only if the range overlaps stored forecasts."""
    rows = data[data["date"].between(start, end)]
    if rows.empty:
        raise ValueError(f"No data between {start} and {end}")
    measured = rows[~rows["target_imputed"]]
    peak = measured["time"].between("07:30", "09:59") | measured["time"].between("17:00", "19:59")
    names = measured.groupby("road_id")["road_name"].first()

    worst = measured.groupby("road_id").agg(severe=("congestion_level", lambda s: (s == "Severe").mean()), speed=("avg_speed", "mean"), volume=("traffic_volume", "mean")).sort_values("severe", ascending=False).head(5).reset_index()
    worst_table = pd.DataFrame({"Segment": worst["road_id"], "Name": worst["road_id"].map(names), "Severe windows": worst["severe"].map("{:.1%}".format), "Mean speed (km/h)": worst["speed"].map("{:.1f}".format), "Mean volume": worst["volume"].map("{:.0f}".format)})

    hotspots = measured.groupby("road_id")["accident_count"].sum().sort_values(ascending=False).head(5).reset_index()
    hotspot_table = pd.DataFrame({"Segment": hotspots["road_id"], "Name": hotspots["road_id"].map(names), "Reported accidents": hotspots["accident_count"]})

    weather = measured.groupby("weather_condition").agg(windows=("traffic_volume", "size"), volume=("traffic_volume", "mean"), speed=("avg_speed", "mean"), accidents=("accident_count", lambda s: (s > 0).mean())).reset_index()
    weather_table = pd.DataFrame({"Weather": weather["weather_condition"], "Windows": weather["windows"].map("{:,}".format), "Mean volume": weather["volume"].map("{:.0f}".format), "Mean speed (km/h)": weather["speed"].map("{:.1f}".format), "Accident windows": weather["accidents"].map("{:.2%}".format)})

    sections = [
        f"# FlowCast summary: {start} to {end}",
        "## Key figures",
        f"- **Windows measured:** {len(measured):,} across {measured['road_id'].nunique()} segments ({len(rows) - len(measured):,} more were estimated and are left out)",
        f"- **Mean volume per window:** {measured['traffic_volume'].mean():.0f} ({measured.loc[peak, 'traffic_volume'].mean():.0f} at peak)",
        f"- **Mean speed:** {measured['avg_speed'].mean():.1f} km/h ({measured.loc[peak, 'avg_speed'].mean():.1f} km/h at peak)",
        f"- **Windows in Heavy or Severe congestion:** {measured['congestion_level'].isin(['Heavy', 'Severe']).mean():.1%}",
        f"- **Reported accidents:** {int(measured['accident_count'].sum()):,}",
        "## Most congested segments",
        _table(worst_table),
        "## Accident hot-spots",
        _table(hotspot_table),
        "## Weather effects (observed)",
        _table(weather_table),
    ]

    in_range = predictions[predictions["target_ts"].dt.strftime("%Y-%m-%d").between(start, end) & predictions["target_valid"]]
    if in_range.empty:
        sections += ["## Forecast accuracy", "No stored forecasts cover this range. Forecasts exist for 16 April to 31 May 2025."]
    else:
        lines = []
        for horizon, part in in_range.groupby("horizon"):
            volume = regression_metrics(part["actual_volume"], part["volume_pred"])
            congestion = congestion_metrics(part["actual_congestion"], part["congestion_pred"])
            has_both = part["actual_accident"].nunique() == 2
            auc = f"{accident_metrics(part['actual_accident'].astype(int), part['accident_score'].to_numpy())['roc_auc']:.3f}" if has_both else "n/a"
            lines.append({"Minutes ahead": horizon * 30, "Forecasts": f"{len(part):,}", "Volume RMSE": f"{volume['rmse']:.1f}", "Volume MAPE": f"{volume['mape_pct']:.1f}%", "Congestion macro-F1": f"{congestion['macro_f1']:.3f}", "Accident ROC-AUC": auc})
        sections += ["## Forecast accuracy for this range", _table(pd.DataFrame(lines))]
    return "\n\n".join(sections) + "\n"


def to_html(summary: str) -> str:
    """Wrap a Markdown summary as a standalone page."""
    body = markdown.markdown(summary, extensions=["tables"])
    style = "body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}th{background:#f2f2f2}"
    return f"<!doctype html><html><head><meta charset='utf-8'><title>FlowCast summary</title><style>{style}</style></head><body>{body}</body></html>"
