import streamlit as st
from common import dataset, page_header, predictions

from src.insights import build_summary, to_html

page_header(
    "Reports and insights",
    "What happened over a period, in a form I can share?",
    "Choose dates for a summary of key figures, the most congested roads, accident hot-spots, weather effects and forecast accuracy. Download it to share.",
)

data = dataset()
days = sorted(data["date"].unique())
start, end = st.select_slider("Dates", options=days, value=(days[-23], days[-1]))
summary = build_summary(data, predictions(include_uploads=False), start, end)

c1, c2 = st.columns(2)
c1.download_button("Download as Markdown", summary, f"flowcast_{start}_{end}.md", "text/markdown", type="primary")
c2.download_button("Download as HTML", to_html(summary), f"flowcast_{start}_{end}.html", "text/html")
st.divider()
st.markdown(summary)
