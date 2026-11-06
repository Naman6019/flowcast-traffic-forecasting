"""Write the three notebooks. `python -m src.pipeline notebooks` then executes them in place."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = """import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src import eda
from src.clean import load_dataset
from src.features import load_features
from src.utils import load_config, project_path

sns.set_theme(style="whitegrid", context="notebook")
cfg = load_config()
FIGURES = project_path("figures", cfg)
df = load_dataset(cfg)
features = load_features(1, cfg)
print(f"{len(df):,} rows, {df['road_id'].nunique()} segments, {df['ts'].min():%d %b %Y} to {df['ts'].max():%d %b %Y}")
"""

EDA = [
    ("md", "# 01: Exploratory data analysis\n\nFlowCast, Northline Corridor. Everything here calls `src/eda.py` on the cleaned dataset, leaving out estimated rows. The last section lists the findings that drove modelling choices."),
    ("code", SETUP),
    ("md", "## Distributions\n\nVolume is right-skewed (skew 1.0) with a long peak-hour tail, and speed is left-skewed (skew -1.0) because congestion pulls it down. Rainfall and visibility are dominated by a dry, clear mode, so rare weather needs its own flags."),
    ("code", """display(eda.distribution_summary(df))
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
rows = eda.observed(df)
for ax, col, label in zip(axes, ["traffic_volume", "avg_speed", "occupancy"], ["Volume (vehicles per window)", "Speed (km/h)", "Occupancy (%)"]):
    sns.histplot(rows[col], bins=60, ax=ax, color="#2a6f97")
    ax.set_xlabel(label)
fig.tight_layout(); fig.savefig(FIGURES / "eda_distributions.png", dpi=120); plt.show()"""),
    ("md", "## Daily and weekly profiles\n\nWeekdays have a sharp morning peak at 08:30 and a broader evening peak around 18:00. Speed mirrors volume. This checks the configured peak windows (07:30-10:00, 17:00-20:00)."),
    ("code", """profile = eda.daily_profile(df)
fig, axes = plt.subplots(1, 2, figsize=(15, 3.8))
for ax, col, label in zip(axes, ["traffic_volume", "avg_speed"], ["Mean volume", "Mean speed (km/h)"]):
    sns.lineplot(profile, x="time", y=col, hue="period", ax=ax)
    ax.set_xticks(ax.get_xticks()[::6]); ax.set_ylabel(label); ax.set_xlabel("Window start")
fig.tight_layout(); fig.savefig(FIGURES / "eda_daily_profile.png", dpi=120); plt.show()
display(profile[profile.period == "Weekday"].nlargest(5, "traffic_volume"))"""),
    ("md", "## Correlation and multicollinearity\n\nThe correlation matrix comes from the from-scratch covariance in `src/scratch.py`. Volume, speed and occupancy are nearly interchangeable (|r| of about 0.9). VIF is the diagonal of the inverse correlation matrix."),
    ("code", """fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(eda.correlation_table(df), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
fig.tight_layout(); fig.savefig(FIGURES / "eda_correlation.png", dpi=120); plt.show()
display(eda.vif_table(features))"""),
    ("md", "## Weather\n\nCompared within peak and off-peak hours, so the clock doesn't confound the result. Note how few fog windows there are at peak (62)."),
    ("code", """weather = eda.weather_effects(df)
display(weather)
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for ax, col, label in zip(axes, ["volume", "speed", "accident_rate_pct"], ["Mean volume", "Mean speed (km/h)", "Accident rate (% of windows)"]):
    sns.barplot(weather, x="weather_condition", y=col, hue="period", ax=ax, order=cfg["weather"]["vocabulary"])
    ax.set_ylabel(label); ax.set_xlabel("")
fig.tight_layout(); fig.savefig(FIGURES / "eda_weather.png", dpi=120); plt.show()"""),
    ("md", "## Holidays, events and roadworks\n\nWeekdays only. These effects are large but rest on very few days, so they can only be used as flags."),
    ("code", "display(eda.calendar_effects(df))"),
    ("md", "## Accidents\n\nThe base rate is under 1% of windows. The tables below show which conditions raise it. `accident_autocorrelation` compares the rate after a recent accident with the base rate (lag 0)."),
    ("code", """accidents = eda.accident_profile(df)
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
sns.barplot(accidents["hour"], x="hour", y="accident_rate_pct", ax=axes[0], color="#bc4749"); axes[0].set_xticks(range(0, 24, 3))
sns.barplot(accidents["weather"], x="weather_condition", y="accident_rate_pct", ax=axes[1], color="#bc4749", order=cfg["weather"]["vocabulary"])
sns.barplot(accidents["vc_band"], x="vc_band", y="accident_rate_pct", ax=axes[2], color="#bc4749")
for ax in axes: ax.set_ylabel("Accident rate (% of windows)")
fig.tight_layout(); fig.savefig(FIGURES / "eda_accidents.png", dpi=120); plt.show()
display(accidents["segment"]["accident_rate_pct"].describe().round(3).to_frame("segment accident rate (%)"))
display(eda.accident_autocorrelation(df))"""),
    ("md", "## Residuals of a naive forecast\n\nThe z-score outlier rule assumes readings that are roughly normal, and regression metrics behave best with even error spread. This looks at the seasonal-naive residual on training dates."),
    ("code", """rows = features[(features.split == "train") & features.target_valid]
residual = rows.y_volume - rows.vol_day
display(pd.Series(eda.naive_residuals(features)).round(2).to_frame("value"))
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
sns.histplot(residual, bins=100, ax=axes[0], stat="density", color="#2a6f97")
grid = np.linspace(residual.quantile(0.001), residual.quantile(0.999), 200)
axes[0].plot(grid, stats.norm.pdf(grid, residual.mean(), residual.std()), color="black", label="Normal, same mean and std"); axes[0].legend()
axes[0].set_xlim(grid[0], grid[-1]); axes[0].set_xlabel("Actual minus same window yesterday")
stats.probplot(residual.sample(5000, random_state=cfg["seed"]), plot=axes[1])
fig.tight_layout(); fig.savefig(FIGURES / "eda_residuals.png", dpi=120); plt.show()"""),
    ("md", "## How much segments move together, and the baselines to beat\n\nCoupling is the median correlation between segments' volume residuals after removing each segment's own window x weekday profile."),
    ("code", """print(f"Median between-segment residual correlation: {eda.segment_coupling(df):.3f}")
display(eda.baseline_scores(features))"""),
    ("md", """## Findings that drive the modelling

1. **Recent values beat seasonal ones.** On validation dates, persistence (RMSE 102, MAPE 16.2%) beats last week (123, 19.2%) and yesterday (139, 23.3%). Lags at t, t-1 and t-2 and the short rolling means should rank highest. These baselines are what every model must beat, and even persistence misses the 12% MAPE target.
2. **Short-term volume features are heavily collinear.** VIF reaches 282 (`vol_mean4`), and is above 40 for the recent lags, V/C and occupancy. That does not hurt prediction, so they stay. It does mean linear coefficients can't be read one at a time, and importance will be shared among them, so the report uses permutation importance on groups of features rather than single coefficients.
3. **Errors are not constant or normal.** The naive residual's spread is more than twice as wide at peak (std 234) as off-peak (107), with excess kurtosis 5.5. This justifies computing the conformal bands separately for peak and off-peak (D12) rather than one width everywhere, and the error breakdowns by hour.
4. **Weather shifts demand more than it slows traffic.** Rain cuts volume by 11% off-peak and 19% at peak; fog by 38% and 35%. Off-peak speed falls about 8% in rain but only about 2% in fog, so **the working hypothesis that fog lowers speed more than rain was wrong**. At peak, speed is actually higher in rain and fog than in clear weather, because lower demand outweighs the slower conditions. Peak fog rests on only 62 windows.
5. **Calendar effects are large but thinly evidenced.** On weekdays, compared with ordinary days, volume is about 45% lower on holidays (2 days), 35% higher on event days (5 days) and 30% lower on roadwork days (9 days). They're used as flags and peak interactions, and the report notes how few days support them.
6. **Accident risk has real but modest signal.** The rate is about 4x higher in rain or fog (about 3.5% against 0.8%), about 3x higher at V/C >= 1 (2.4% against 0.75%), and higher at the morning peak. It varies little by segment (0.67-1.19%). A recent accident raises the next window's rate from 0.91% to 1.63%, and this fades within about two windows. So `accidents_last4`, weather and interaction features are justified. But the target-hour weather isn't known at issue time, which limits what a model can reach: the ROC-AUC >= 0.75 target remains the riskiest.
7. **Congestion is imbalanced and rule-based.** Free-flow 61%, Moderate 24%, Heavy 9%, Severe 5%. The label is a fixed function of volume, so regress-then-band is a strong candidate, and direct classifiers need class weights.
8. **Segments move together.** The median residual correlation between segments is 0.64, even after removing each segment's own profile, which points to shared corridor-wide day effects. One global model across all segments is therefore well justified, for both the classical and deep models.
"""),
]

CLEANING = [
    ("md", "# 02: Cleaning walkthrough\n\nThe pipeline itself is `src/ingest.py` and `src/clean.py`. This notebook audits its output using the quarantine ledger, where every rejected row and changed value is recorded. The full generated report is `reports/data_quality.md`."),
    ("code", """import sys
sys.path.insert(0, "..")
import matplotlib.pyplot as plt
import pandas as pd

from src.clean import load_dataset
from src.utils import load_config, project_path

cfg = load_config()
df = load_dataset(cfg)
ledger = pd.read_parquet(project_path("interim", cfg) / "quarantine.parquet")
print(f"{len(ledger):,} ledger entries, {len(df):,} rows in the final dataset")"""),
    ("md", "## What each rule did"),
    ("code", """display(ledger.groupby(["stage", "rule", "column"], sort=False).size().rename("entries").reset_index())"""),
    ("md", "## Repairs are exact where a redundant column exists\n\nVolume rebuilt from `vehicle_count` and speed rebuilt from `travel_time` are compared with the values that survive on valid rows."),
    ("code", """sensor = df[df.vol_source == "sensor"]
print("Sensor rows where volume != vehicle_count:", int((sensor.traffic_volume != sensor.vehicle_count).sum()))
valid = df[df.speed_source == "sensor"]
implied = valid.seg_length_km * 60 / valid.travel_time
print(f"Speed implied by travel time vs sensor speed on valid rows: max abs error {(implied - valid.avg_speed).abs().max():.2f} km/h, mean {(implied - valid.avg_speed).abs().mean():.3f} km/h")"""),
    ("md", "## One repaired stretch\n\nA day containing an absent window: the markers show rows the pipeline estimated."),
    ("code", """row = df[df.row_imputed & (df.date > "2025-01-05")].iloc[0]
day = df[(df.road_id == row.road_id) & (df.date == row.date)]
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(day.time, day.traffic_volume, color="#2a6f97", label="Cleaned volume")
gaps = day[day.row_imputed]
ax.scatter(gaps.time, gaps.traffic_volume, color="#bc4749", zorder=3, label="Estimated (absent window)")
ax.set_xticks(ax.get_xticks()[::6]); ax.set_title(f"{row.road_id} on {row.date}"); ax.legend()
plt.show()"""),
    ("md", "## Final checks\n\nThe pipeline fails rather than continue if any of these break. They're re-run here against the saved dataset."),
    ("code", """from src.clean import final_checks
final_checks(df, 181_200, cfg)
print("Passed: no nulls in modelling columns, 181,200 rows, unique keys, ranges and vocabulary valid")
display(df.congestion_level.value_counts().to_frame("windows"))"""),
]


MODELLING = [
    ("md", "# 03: Modelling walkthrough\n\nThe engines are `src/ml_models.py` and `src/dl_model.py`, and `src/final.py` is the only code that scores the test window. This notebook reads their saved outputs. The one exception is the gradient-descent demonstration, which fits on training rows only."),
    ("code", """import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

from src import scratch
from src.evaluate import split_rows
from src.features import feature_columns, load_features
from src.ml_models import make_preprocessor
from src.utils import load_config

cfg = load_config()
REPORTS = Path("../reports")
metrics = json.loads((REPORTS / "final_metrics.json").read_text(encoding="utf-8"))"""),
    ("md", "## Results against the PRD targets"),
    ("code", """display(pd.DataFrame(metrics["criteria"]).round(3))"""),
    ("md", "## Gradient descent, written in NumPy\n\nThe PRD's build-it-once rule. The cost is J(w) = (1/n) ||Xw - y||^2, with gradient (2/n) X^T (Xw - y). Features are heavily collinear, so a fixed step like 0.1 diverges. The step comes from the largest eigenvalue of X^T X instead, which guarantees descent. The closed-form solution is the reference, and progress is slow along the small-eigenvalue directions that collinearity creates."),
    ("code", """frame = load_features(1, cfg)
columns = feature_columns(frame)
train = split_rows(frame, "train")
X = make_preprocessor(columns).fit_transform(train[columns])
y = train["y_volume"].to_numpy()
weights, history = scratch.gradient_descent_lr(X, y, epochs=cfg["tuning"]["scratch_epochs"])
exact = scratch.normal_equation(X, y)
print(f"Training MSE: gradient descent {history[-1]:.1f}, normal equation {scratch.mse(y, scratch.predict_linear(X, exact)):.1f}")
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.semilogy(history, color="#2a6f97"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (log scale)"); ax.set_title("Linear regression trained by batch gradient descent")
fig.tight_layout(); fig.savefig(REPORTS / "figures" / "scratch_gradient_descent.png", dpi=120); plt.show()"""),
    ("md", "## Scoreboards\n\nEvery model on the same rows. CV and validation scores drove the choices, and test scores are for the record."),
    ("code", """for target, metric in [("volume", "rmse"), ("travel_time", "rmse"), ("speed", "rmse"), ("congestion", "macro_f1"), ("accident", "roc_auc")]:
    board = pd.read_csv(REPORTS / f"scoreboard_{target}.csv")
    keep = [c for c in ["model", "cv_mean", "cv_std", f"val_{metric}", f"test_{metric}", "infer_ms_per_1k"] if c in board]
    display(Markdown(f"**{target.replace('_', ' ').title()}**"), board[keep].round(4))"""),
    ("md", "## Classical vs deep"),
    ("code", """display(Markdown((REPORTS / "benchmark.md").read_text(encoding="utf-8")))
display(Image(REPORTS / "figures" / "deep_loss_curves.png"))"""),
    ("md", "## Error analysis (PRD 13.3)"),
    ("code", """display(Image(REPORTS / "figures" / "final_volume_residuals.png"))
errors = pd.read_csv(REPORTS / "error_breakdown_volume.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
for ax, group in zip(axes, ["hour", "day_of_week", "weather"]):
    part = errors[errors["group"] == group]
    if group == "hour":
        part = part.assign(value=part["value"].astype(int)).sort_values("value")
    if group == "day_of_week":
        part = part.set_index("value").reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]).reset_index()
    ax.bar(part["value"].astype(str), part["rmse"], color="#2a6f97"); ax.set_title(f"Test RMSE by {group.replace('_', ' ')}"); ax.tick_params(axis="x", labelsize=8)
fig.tight_layout(); plt.show()
display(pd.read_csv(REPORTS / "confusion_congestion.csv", index_col=0))"""),
    ("md", "## Accuracy by look-ahead, and whether the bands hold up"),
    ("code", """display(pd.read_csv(REPORTS / "metrics_by_horizon.csv").T.round(2))"""),
    ("md", "## What the models rely on"),
    ("code", """display(pd.read_csv(REPORTS / "feature_importance_groups.csv").round(4))"""),
]


def write(name: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook(metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    notebook.cells = [nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text) for kind, text in cells]
    nbf.write(notebook, ROOT / "notebooks" / name)


if __name__ == "__main__":
    write("01_eda.ipynb", EDA)
    write("02_cleaning.ipynb", CLEANING)
    write("03_modelling.ipynb", MODELLING)
