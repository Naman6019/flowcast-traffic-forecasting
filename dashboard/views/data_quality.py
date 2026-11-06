import pandas as pd
import streamlit as st
from common import CONFIG, REPORTS, dataset, page_header, road_names, takeaway

from src.utils import project_path

page_header(
    "Data quality",
    "What was wrong with the raw data, and what was changed?",
    "Real sensor feeds have gaps and faults, and this data has the same kinds of problems. Nothing is dropped or changed silently: every fix is logged with its old and new value.",
)

ledger = pd.read_parquet(project_path("interim", CONFIG) / "quarantine.parquet")
data = dataset()
counts = ledger.groupby("rule").size()


def count(*rules: str) -> int:
    return int(sum(counts.get(rule, 0) for rule in rules))


duplicates, added = count("duplicate_key"), count("row_added_to_grid")
received = len(data) - added + duplicates + int((ledger["column"].eq("row") & ledger["stage"].eq("ingest")).sum())
repaired = count("volume_from_vehicle_count", "speed_from_travel_time")
takeaway(f"Of {received:,} rows received, {duplicates:,} were duplicates, and {added:,} expected readings were missing altogether. Those estimated periods can feed a forecast as input, but they're never used to score one, so the accuracy figures only count real measurements.")

a, b, c, d = st.columns(4)
a.metric("Rows received", f"{received:,}")
b.metric("Rows in the final dataset", f"{len(data):,}")
c.metric("Faulty readings rebuilt exactly", f"{repaired:,}")
d.metric("Missing periods estimated", f"{added:,}")

st.subheader("Problems found, and how each was fixed")
problems = [
    ("Same reading logged twice", "duplicate_key", "Kept the most complete copy."),
    ("No reading for a 30-minute period", "row_added_to_grid", "Estimated from the neighbouring readings. Never used to score forecasts."),
    ("Vehicle count blank or negative", "volume_from_vehicle_count", "Rebuilt from the raw detector count, which matches on every valid row."),
    ("Speed blank or impossible (above 200 km/h)", "speed_from_travel_time", "Rebuilt from travel time and road length."),
    ("Road occupancy above 100%", "occupancy_blanked", "Removed, then estimated from neighbouring readings."),
    ("Congestion level blank", "congestion_filled", "Worked out from the published rule, which matches every labelled row."),
    ("Weather labels written inconsistently, like 'rainy' and 'RAIN'", "weather_label_normalised", "Mapped to Clear, Cloudy, Rain or Fog."),
    ("Temperature, visibility or rainfall blank", "weather_interpolated", "Estimated from the same station's neighbouring hours."),
]
table = pd.DataFrame([{"Problem": name, "How many": count(rule), "What was done": fix} for name, rule, fix in problems])
st.dataframe(table.style.format({"How many": "{:,}"}), hide_index=True, width="stretch")
flagged = int(data["stat_outlier"].sum())
st.caption(f"A further {flagged:,} unusual but possible readings ({flagged / len(data):.1%}) are flagged and left unchanged, because traffic has real spikes. Weather dates (DD/MM/YYYY) and traffic dates (YYYY-MM-DD) are parsed explicitly, so they can't be confused.")

st.subheader("Look up the changes")
st.caption("Every entry in the change log. Choose a kind of change to see the rows it affected.")
rule_names = {rule: name for name, rule, _ in problems}
available = list(ledger["rule"].value_counts().index)
rule = st.selectbox("Kind of change", available, index=available.index("speed_from_travel_time") if "speed_from_travel_time" in available else 0, format_func=lambda r: f"{rule_names.get(r, r.replace('_', ' ').capitalize())} ({counts[r]:,})")
entries = ledger[ledger["rule"] == rule]
names = road_names()
parts = entries["key"].str.split("|", n=1, expand=True)
view = pd.DataFrame({"Road or station": parts[0].map(lambda r: names.get(r, r)), "When": parts[1] if parts.shape[1] > 1 else "", "Field": entries["column"].to_numpy(), "Old value": entries["old_value"].fillna("blank").to_numpy(), "New value": entries["new_value"].fillna("-").to_numpy()})
st.dataframe(view.head(500), hide_index=True, width="stretch", height=300)
st.caption(f"Showing the first {min(500, len(view)):,} of {len(view):,} entries.")
st.download_button("Download these entries", entries.to_csv(index=False), f"changes_{rule}.csv", "text/csv")

with st.expander("Full data-quality report"):
    st.markdown((REPORTS / "data_quality.md").read_text(encoding="utf-8"))
