import json

import pandas as pd
import streamlit as st
from common import FEATURE_NAMES, MODEL_NAMES, ROOT, TARGET_NAMES, page_header, registry

page_header(
    "Model records",
    "Which model produced a forecast, and how was it built?",
    "Every model has a record of what it learned from, its settings and its scores, so any forecast can be traced back and reproduced.",
)

reg = registry()
production = reg["production"]
st.subheader("In service now")
congestion = production["congestion"]
service = pd.DataFrame(
    [
        ("Traffic volume", MODEL_NAMES[reg["deep"]["model"]] if production["volume"] == "deep" else MODEL_NAMES[reg["targets"]["volume"]["model"]], "Lowest typical error on validation dates"),
        ("Speed", MODEL_NAMES[reg["targets"]["speed"]["model"]], "Same type as the best non-neural volume model"),
        ("Travel time", MODEL_NAMES[reg["targets"]["travel_time"]["model"]], "Lowest typical error on validation dates"),
        ("Congestion level", f"{MODEL_NAMES[congestion['model']]} ({', '.join(str(c) for c in congestion['cuts'])})", "Best validation score, and it stays consistent with the volume shown"),
        ("Accident risk", MODEL_NAMES[reg["targets"]["accident"]["model"]], "Best ranking on validation dates, and simple to read"),
    ],
    columns=["Prediction", "Model", "Why it was chosen"],
)
st.dataframe(service, hide_index=True, width="stretch")
st.caption("Each prediction has one model for every look-ahead: 30, 60, 90 and 120 minutes.")


@st.cache_data
def load_cards() -> list:
    cards = []
    for path in sorted((ROOT / "models").rglob("model_card.json")):
        if "retrain" in path.as_posix():
            continue
        card = json.loads(path.read_text(encoding="utf-8"))
        card["path"] = path.parent.relative_to(ROOT).as_posix()
        cards.append(card)
    return cards


cards = load_cards()
st.subheader("Look up a model")
st.caption(f"{len(cards)} model records.")
c1, c2, c3 = st.columns(3)
targets = [t for t in TARGET_NAMES if any(card["target"] == t for card in cards)]
target = c1.selectbox("Prediction", targets, format_func=TARGET_NAMES.get)
families = sorted({card["model"] for card in cards if card["target"] == target})
family = c2.selectbox("Model", families, format_func=lambda m: MODEL_NAMES.get(m, m))
horizons = sorted(card["horizon"] for card in cards if card["target"] == target and card["model"] == family)
horizon = c3.selectbox("Minutes ahead", horizons, format_func=lambda h: str(h * 30))
card = next(c for c in cards if c["target"] == target and c["model"] == family and c["horizon"] == horizon)

a, b, c = st.columns(3)
a.metric("Trained on", f"{card['training_window'][0]} to {card['training_window'][1]}")
b.metric("Inputs used", len(card["features"]))
c.metric("Random seed", card["random_seed"])

left, right = st.columns(2)
scores = pd.DataFrame({"Validation dates": card.get("validation_metrics") or {}, "Test dates": card.get("test_metrics") or {}})
scores = scores[~scores.index.str.startswith(("f1_class", "base_rate"))]
left.markdown("**Scores**")
left.dataframe(scores.style.format("{:.4f}", na_rep="-"), width="stretch")
right.markdown("**Settings**")
settings = card.get("hyperparameters") or {}
if settings:
    right.dataframe(pd.DataFrame({"Value": {k: str(v) for k, v in settings.items()}}), width="stretch")
else:
    right.write("This model has no settings to tune.")

with st.expander("Inputs this model uses"):
    st.dataframe(pd.DataFrame({"Input": [FEATURE_NAMES.get(f, f) for f in card["features"]], "Column": card["features"]}), hide_index=True, width="stretch")
with st.expander("Where it's stored and how to reproduce it"):
    st.markdown(f"- **Folder:** `{card['path']}`\n- **Dataset fingerprint:** `{card['dataset_sha256'][:16]}`\n- **Code version:** `{card['git_commit'][:10]}`")
    if card.get("trained_on"):
        st.markdown(f"- **Trained on device:** {card['trained_on']}")
    if card.get("pretrained_weights") is False:
        st.markdown("- **Pretrained weights:** none. It was trained from scratch.")
