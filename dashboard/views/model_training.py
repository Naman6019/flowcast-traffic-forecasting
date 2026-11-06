import json
import subprocess
import sys

import pandas as pd
import streamlit as st
from common import MODEL_NAMES, REPORTS, ROOT, page_header, registry

page_header(
    "Retrain a model",
    "Does a freshly trained model match the one in use?",
    "This trains a new copy of a model with its saved settings, then compares it with the one in use. It only runs when you press the button, and the new copy doesn't replace the current model.",
)

targets = {"volume": "Traffic volume", "speed": "Speed", "accident": "Accident risk"}
target = st.selectbox("Model", list(targets), format_func=targets.get)
entry = registry()["targets"][target]
st.markdown(f"Current model: **{MODEL_NAMES.get(entry['model'], entry['model'])}**")
if target == "volume" and registry()["production"]["volume"] == "deep":
    st.caption("Volume forecasts on this dashboard come from a neural network, which takes minutes to retrain. This trains the best non-neural model, which takes seconds.")

if st.button("Start retraining", type="primary"):
    log = st.empty()
    lines = []
    process = subprocess.Popen([sys.executable, "-m", "src.retrain", "--target", target], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        lines.append(line.rstrip())
        log.code("\n".join(lines[-15:]))
    if process.wait() != 0:
        st.error("Retraining failed. The model in use hasn't changed, and the log above shows the error.")
        st.stop()
    st.session_state["retrained"] = True

path = REPORTS / "last_retrain.json"
if st.session_state.get("retrained") and path.exists():
    result = json.loads(path.read_text(encoding="utf-8"))[0]
    st.success(f"Finished in {result['seconds']} s. The new copy is saved at {result['new_path']}.")
    table = pd.DataFrame({"In use": result["current_validation"], "New copy": result["new_validation"]})
    table["Change"] = table["New copy"] - table["In use"]
    st.subheader("Scores on validation dates")
    st.dataframe(table.style.format("{:.4f}"), width="stretch")
    st.caption("With the same data, settings and seed, the two should match. A difference points to a change in the data or software.")
